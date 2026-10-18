"""Shared pieces: the prompt, model loading, generation, JSON parsing and scoring."""
import json
import pathlib
import re

import torch

ROOT = pathlib.Path(__file__).parent
DATA = ROOT / "data"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

ASSAY_VOCAB = ["RNA-seq", "single-cell RNA-seq", "qPCR", "Western blot", "ELISA",
               "flow cytometry", "immunohistochemistry", "ChIP-seq", "mass spectrometry",
               "calcium imaging", "patch-clamp electrophysiology"]

# The same system prompt is used for zero-shot, few-shot AND the fine-tuned model,
# so the comparison isolates the effect of the weights (or of the examples).
SYSTEM_PROMPT = f"""You extract structured data from the methods sections of biology papers.
Return ONLY a JSON object with exactly these keys:
  "organism": the species studied, as a Latin binomial (e.g. mice -> "Mus musculus"; human cell lines such as HeLa -> "Homo sapiens"). Ignore species that are only antibody hosts.
  "assay": the main experimental assay, chosen from this list: {json.dumps(ASSAY_VOCAB)}
  "sample_size": total number of subjects or biological replicates as an integer. If given per group, multiply by the number of groups. Use null if not stated. Technical replicates do not count.
No explanation, no markdown."""


def load_jsonl(name):
    with open(DATA / f"{name}.jsonl") as fh:
        return [json.loads(line) for line in fh]


def build_messages(text, few_shot=None):
    """Chat messages for one example. few_shot: list of training rows used as worked examples."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in few_shot or []:
        msgs.append({"role": "user", "content": ex["text"]})
        msgs.append({"role": "assistant", "content": json.dumps(ex["label"])})
    msgs.append({"role": "user", "content": text})
    return msgs


def target_json(label):
    """The exact string the model is trained to produce (fixed key order)."""
    return json.dumps({"organism": label["organism"], "assay": label["assay"],
                       "sample_size": label["sample_size"]})


def load_model(adapter=None):
    """Load base model + tokenizer on CPU in fp32; optionally attach a LoRA adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.padding_side = "left"   # left-pad so batched generation continues from real tokens
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


@torch.no_grad()
def generate(tok, model, message_lists, max_new_tokens=64, batch_size=8):
    """Greedy-decode a list of chat conversations. Returns the decoded completions."""
    outs = []
    for i in range(0, len(message_lists), batch_size):
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                   for m in message_lists[i:i + batch_size]]
        enc = tok(prompts, return_tensors="pt", padding=True)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for row in gen[:, enc["input_ids"].shape[1]:]:
            outs.append(tok.decode(row, skip_special_tokens=True))
    return outs


def parse_json(raw):
    """Return (dict or None, strict). strict=True if the whole output is a JSON object."""
    s = raw.strip()
    try:
        obj = json.loads(s)
        return (obj, True) if isinstance(obj, dict) else (None, False)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*?\}", s, re.S)   # lenient: first {...} anywhere (e.g. inside ```json fences)
    if m:
        try:
            obj = json.loads(m.group(0))
            return (obj, False) if isinstance(obj, dict) else (None, False)
        except json.JSONDecodeError:
            pass
    return None, False


def _norm(x):
    return str(x).strip().lower() if x is not None else None


def score(pred, label):
    """Per-field correctness for one prediction (pred may be None if parsing failed)."""
    pred = pred or {}
    ss = pred.get("sample_size")
    if isinstance(ss, str) and ss.strip().isdigit():
        ss = int(ss)
    r = {
        "organism": _norm(pred.get("organism")) == _norm(label["organism"]),
        "assay": _norm(pred.get("assay")) == _norm(label["assay"]),
        "sample_size": ss == label["sample_size"],
    }
    r["all_fields"] = all(r.values())
    return r
