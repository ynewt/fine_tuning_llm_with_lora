"""Fine-tune Qwen2.5-0.5B-Instruct with LoRA on the methods->JSON task (CPU, fp32).

A plain PyTorch loop (no Trainer) so every step is visible:
  1. wrap the frozen base model with LoRA adapters (PEFT)
  2. tokenize prompt + target JSON; mask the prompt so loss is only on the answer
  3. AdamW + linear warmup/decay, gradient accumulation
  4. log train loss every step and val loss every --eval_every steps
  5. save the adapter after every epoch (adapters/<run>/epoch<k>/)

Usage:
  python train_lora.py                                  # defaults: 300 examples, 2 epochs
  python train_lora.py --n_train 40 --epochs 8 --run overfit   # overfitting demo
"""
import argparse
import json
import math
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from common import BASE_MODEL, ROOT, build_messages, load_jsonl, target_json


def encode(tok, ex, max_len):
    """input_ids for [chat prompt + JSON answer + <|im_end|>]; labels = -100 on the prompt."""
    prompt = tok.apply_chat_template(build_messages(ex["text"]), tokenize=False,
                                     add_generation_prompt=True)
    answer = target_json(ex["label"]) + "<|im_end|>"
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(answer, add_special_tokens=False)["input_ids"]
    ids = (p_ids + a_ids)[:max_len]
    labels = ([-100] * len(p_ids) + a_ids)[:max_len]
    return torch.tensor([ids]), torch.tensor([labels])


@torch.no_grad()
def val_loss(model, batches):
    model.eval()
    tot = sum(model(input_ids=i, labels=l).loss.item() for i, l in batches)
    model.train()
    return tot / len(batches)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main", help="name for adapters/<run> and runs/<run>_loss.jsonl")
    ap.add_argument("--n_train", type=int, default=None, help="use only the first N training rows")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--grad_accum", type=int, default=8, help="examples per optimizer step")
    ap.add_argument("--eval_every", type=int, default=5, help="optimizer steps between val-loss checks")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)

    # --- LoRA: freeze everything, add rank-r update matrices to attention + MLP projections
    cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"],
                     task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    model.train()

    train = load_jsonl("train")[:args.n_train]
    val = [encode(tok, ex, args.max_len) for ex in load_jsonl("val")]
    print(f"train examples: {len(train)}   val examples: {len(val)}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                            weight_decay=0.0)
    total_steps = math.ceil(len(train) / args.grad_accum) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, max(1, int(0.1 * total_steps)), total_steps)

    out_dir = ROOT / "adapters" / args.run
    log_path = ROOT / "runs" / f"{args.run}_loss.jsonl"
    log_path.parent.mkdir(exist_ok=True)
    log = open(log_path, "w")
    log.write(json.dumps({"type": "config", **vars(args), "total_steps": total_steps}) + "\n")

    def write(rec):
        log.write(json.dumps(rec) + "\n")
        log.flush()

    v = val_loss(model, val)
    write({"type": "val", "step": 0, "epoch": 0.0, "loss": v})
    print(f"step   0  val_loss {v:.4f}  (before training)")

    step, t0 = 0, time.time()
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train)))
        random.shuffle(order)
        running = []
        for j, idx in enumerate(order):
            ids, labels = encode(tok, train[idx], args.max_len)
            loss = model(input_ids=ids, labels=labels).loss
            (loss / args.grad_accum).backward()      # accumulate gradients over several examples
            running.append(loss.item())
            last = j == len(order) - 1
            if (j + 1) % args.grad_accum == 0 or last:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                ep = epoch - 1 + (j + 1) / len(order)
                tl = sum(running) / len(running)
                write({"type": "train", "step": step, "epoch": ep, "loss": tl,
                       "lr": sched.get_last_lr()[0]})
                running = []
                msg = f"step {step:3d}/{total_steps}  ep {ep:.2f}  train_loss {tl:.4f}"
                if step % args.eval_every == 0 or last:
                    v = val_loss(model, val)
                    write({"type": "val", "step": step, "epoch": ep, "loss": v})
                    msg += f"  val_loss {v:.4f}"
                print(msg + f"  [{(time.time() - t0) / 60:.1f} min]", flush=True)
        model.save_pretrained(out_dir / f"epoch{epoch}")
        print(f"saved adapter -> {out_dir / f'epoch{epoch}'}")
    log.close()


if __name__ == "__main__":
    main()
