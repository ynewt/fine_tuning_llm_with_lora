"""Generate a synthetic "messy methods section -> JSON" dataset.

Every example is assembled from sentence templates, so the ground truth is known
exactly. Messiness comes from distractor sentences (reagents, centrifuge speeds,
antibody host species, ages, ethics protocol numbers), random sentence order,
citation brackets, line-break hyphenation and stray whitespace.

Splits written to data/:
  train.jsonl     in-distribution templates and organisms
  val.jsonl       same distribution as train; used for the val-loss curve
  test_id.jsonl   same distribution, never trained on
  test_ood.jsonl  templates AND organisms the model never saw in training

The gap between test_id and test_ood measures how much of what fine-tuning
learned is the task itself and how much is just the surface form of the
training set.

Usage:  python make_dataset.py [--n_train 300 --n_val 40 --n_test 80 --seed 0]
"""
import argparse
import json
import pathlib
import random

DATA = pathlib.Path(__file__).parent / "data"

# ---------------------------------------------------------------------------
# Label vocabularies
# ---------------------------------------------------------------------------
# organism: canonical binomial -> (kind, surface mentions, subject noun)
#   kind decides how sample size is phrased (animals vs. participants vs. replicates)
ORGANISMS_ID = {
    "Mus musculus": ("animal", ["C57BL/6J mice", "BALB/c mice", "male C57BL/6 mice",
                                "Ai14 reporter mice", "8-week-old mice"], "mice"),
    "Rattus norvegicus": ("animal", ["Sprague-Dawley rats", "Wistar rats",
                                     "adult male Long-Evans rats"], "rats"),
    "Homo sapiens": ("human", ["participants", "patients with type 2 diabetes",
                               "healthy volunteers", "adults with mild asthma"], "participants"),
    "Danio rerio": ("animal", ["zebrafish larvae", "AB-strain zebrafish",
                               "transgenic Tg(fli1:EGFP) zebrafish embryos"], "fish"),
    "Drosophila melanogaster": ("animal", ["w1118 flies", "female Drosophila",
                                           "Canton-S fruit flies"], "flies"),
    "Caenorhabditis elegans": ("animal", ["N2 hermaphrodites", "C. elegans (N2 Bristol)",
                                          "synchronized L4 worms"], "worms"),
    "Saccharomyces cerevisiae": ("culture", ["S. cerevisiae BY4741", "budding yeast (W303)",
                                             "yeast strain BY4742"], "cultures"),
    "Escherichia coli": ("culture", ["E. coli MG1655", "E. coli K-12",
                                     "E. coli BL21(DE3)"], "cultures"),
    "Arabidopsis thaliana": ("animal", ["Arabidopsis Col-0 seedlings",
                                        "A. thaliana plants (Col-0)"], "plants"),
}
# Human cell lines: organism is still Homo sapiens. This is a convention the
# model has to learn ("HEK293T cells" -> Homo sapiens).
HUMAN_CELL_LINES = ["HEK293T cells", "HeLa cells", "MCF-7 cells", "A549 cells"]

# Held-out organisms (test_ood only). The base model knows these binomials from
# pretraining; a fine-tuned model has never been asked to produce them.
ORGANISMS_OOD = {
    "Oryctolagus cuniculus": ("animal", ["New Zealand White rabbits", "adult rabbits"], "rabbits"),
    "Sus scrofa": ("animal", ["Yorkshire pigs", "juvenile domestic pigs"], "pigs"),
    "Xenopus laevis": ("animal", ["Xenopus laevis tadpoles", "X. laevis frogs"], "frogs"),
    "Gallus gallus": ("animal", ["broiler chickens", "White Leghorn chickens"], "birds"),
}

ASSAYS = {
    "RNA-seq": ["bulk RNA sequencing", "RNA-seq", "transcriptome sequencing on an Illumina NovaSeq",
                "poly(A)-selected mRNA sequencing"],
    "single-cell RNA-seq": ["scRNA-seq", "10x Genomics Chromium single-cell RNA sequencing",
                            "droplet-based single-cell transcriptomics"],
    "qPCR": ["qRT-PCR", "real-time quantitative PCR", "RT-qPCR (SYBR Green)"],
    "Western blot": ["western blotting", "immunoblotting", "immunoblot analysis"],
    "ELISA": ["ELISA", "enzyme-linked immunosorbent assay", "sandwich ELISA"],
    "flow cytometry": ["flow cytometry", "FACS analysis", "cytometric analysis on a BD LSRFortessa"],
    "immunohistochemistry": ["IHC", "immunohistochemical staining",
                             "immunostaining of paraffin-embedded sections"],
    "ChIP-seq": ["ChIP-seq", "chromatin immunoprecipitation followed by sequencing"],
    "mass spectrometry": ["LC-MS/MS", "label-free quantitative proteomics",
                          "tandem mass spectrometry"],
    "calcium imaging": ["two-photon calcium imaging", "GCaMP6s imaging", "Fluo-4 calcium imaging"],
    "patch-clamp electrophysiology": ["whole-cell patch-clamp recordings",
                                      "patch-clamp electrophysiology"],
}
ASSAY_VOCAB = list(ASSAYS)

NUM_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 8: "eight", 10: "ten",
             12: "twelve", 16: "sixteen", 20: "twenty", 24: "twenty-four"}

# ---------------------------------------------------------------------------
# Sentence templates. Each list is split into in-distribution and OOD templates.
# {org} = organism mention, {subj} = subject noun, {assay} = assay phrase
# ---------------------------------------------------------------------------
SUBJECT_TOTAL_ID = [
    "A total of {n} {org} were used in this study.",
    "We studied {n} {org}.",
    "{N} {org} were included (n = {n}).",
    "Experiments were performed on {org} (n = {n} in total).",
]
SUBJECT_TOTAL_OOD = [
    "Our cohort comprised {n} {org}, all of which completed the protocol.",
    "In all, {n} {org} entered the analysis set.",
]
SUBJECT_PER_GROUP_ID = [
    "{org} were randomly assigned to {k} groups (n = {m} per group).",
    "{org} were divided into {k_word} treatment arms of {m} {subj} each.",
]
SUBJECT_PER_GROUP_OOD = [
    "Each of the {k} experimental conditions contained {m} {subj} ({org}).",
]
SUBJECT_NULL_ID = [
    "{org} were maintained under standard conditions.",
    "Experiments used {org} obtained from a commercial supplier.",
    "All procedures were carried out on {org}.",
]
SUBJECT_NULL_OOD = [
    "The study population consisted of {org}; group sizes are reported in the figure legends.",
]
# Cell / culture systems phrase sample size as biological replicates.
REPLICATE_TOTAL_ID = [
    "{org} were cultured and assayed in {n} independent biological replicates.",
    "All experiments with {org} were performed in biological {rep_word}.",
]
REPLICATE_TOTAL_OOD = [
    "Data from {org} represent {n} separately grown cultures.",
]
REPLICATE_PER_GROUP_ID = [
    "{org} were exposed to {k} conditions, with {m} biological replicates per condition.",
]
REPLICATE_PER_GROUP_OOD = [
    "For {org}, {m} replicate cultures were prepared for each of {k} treatments.",
]
REPLICATE_NULL_ID = [
    "{org} were grown to mid-log phase before harvesting.",
    "{org} were obtained from ATCC and maintained at 37 °C.",
]
REPLICATE_NULL_OOD = [
    "{org} were propagated as previously described [{cite}].",
]
ASSAY_ID = [
    "Samples were analyzed by {assay}.",
    "{Assay} was performed as described previously [{cite}].",
    "The primary readout was {assay}.",
    "We then carried out {assay} on harvested material.",
]
ASSAY_OOD = [
    "Downstream measurements relied on {assay}, following the manufacturer's protocol.",
    "Our main endpoint was quantified using {assay}.",
]

# Distractors: numbers and organism names that are NOT the answer.
DISTRACTORS = [
    "Tissue was homogenized and centrifuged at {g} × g for {mins} min at 4 °C.",
    "Primary antibodies were rabbit anti-GAPDH (1:{dil}, Cell Signaling #{cat}) and goat anti-mouse IgG.",
    "Animals and cells were handled under protocol #{year}-{proto} approved by the institutional committee.",
    "Statistical analyses were performed in GraphPad Prism 9; p < 0.05 was considered significant.",
    "Reagents were purchased from Sigma-Aldrich (St. Louis, MO) unless noted otherwise.",
    "Incubation lasted {hours} h at {temp} °C in a humidified chamber.",
    "Data are presented as mean ± SEM of {tech} technical replicates.",
    "Images were acquired on a Zeiss LSM {lsm} confocal microscope with a {obj}x objective.",
    "RNA was extracted with TRIzol (Invitrogen, cat. {cat}) and quantified on a NanoDrop.",
    "Buffer contained {mm} mM Tris-HCl, 150 mM NaCl and 0.1% Tween-20.",
]


def rand_fill(rng):
    return dict(g=rng.choice([500, 1000, 12000, 14000, 16000]), mins=rng.choice([5, 10, 15, 20]),
                dil=rng.choice([500, 1000, 2000, 5000]), cat=rng.randint(1000, 99999),
                year=rng.choice([2018, 2019, 2020, 2021, 2022]), proto=rng.randint(10, 999),
                hours=rng.choice([1, 2, 4, 12, 24, 48]), temp=rng.choice([4, 25, 30, 37]),
                tech=rng.choice([2, 3, 4]), lsm=rng.choice([710, 780, 880, 980]),
                obj=rng.choice([10, 20, 40, 63]), mm=rng.choice([20, 25, 50]),
                cite=rng.randint(1, 60))


def messify(text, rng):
    """Add the kind of noise you get from PDF copy-paste."""
    words = text.split(" ")
    out = []
    for w in words:
        r = rng.random()
        if r < 0.02 and len(w) > 7 and w.isalpha():
            cut = rng.randint(3, len(w) - 3)
            out.append(w[:cut] + "-\n" + w[cut:])        # line-break hyphenation
        elif r < 0.04:
            out.append(w + " ")                           # double space
        else:
            out.append(w)
    text = " ".join(out)
    if rng.random() < 0.3:
        text = text.replace(". ", ".\n", rng.randint(1, 2))  # stray line breaks
    return text


def pick(rng, id_list, ood_list, ood):
    return rng.choice(ood_list if ood else id_list)


def make_example(rng, ood):
    orgs = ORGANISMS_OOD if ood else ORGANISMS_ID
    organism = rng.choice(list(orgs))
    kind, mentions, subj = orgs[organism]
    mention = rng.choice(mentions)
    if organism == "Homo sapiens" and rng.random() < 0.4:
        kind, mention, subj = "culture", rng.choice(HUMAN_CELL_LINES), "cultures"

    assay = rng.choice(ASSAY_VOCAB)
    assay_phrase = rng.choice(ASSAYS[assay])

    f = rand_fill(rng)
    mode = rng.choices(["total", "per_group", "null"], weights=[0.45, 0.35, 0.20])[0]
    is_cells = kind == "culture"
    if mode == "total":
        if is_cells:
            tpl = pick(rng, REPLICATE_TOTAL_ID, REPLICATE_TOTAL_OOD, ood)
            n = rng.choice([3, 4] if "{rep_word}" in tpl else [3, 4, 5, 6])
        else:
            tpl = pick(rng, SUBJECT_TOTAL_ID, SUBJECT_TOTAL_OOD, ood)
            n = rng.choice([6, 8, 10, 12, 16, 20, 24, 30, 36, 40, 48, 60, 120, 250])
        sample_size = n
        subj_sent = tpl.format(org=mention, subj=subj, n=n, N=NUM_WORDS.get(n, str(n)).capitalize(),
                               rep_word={3: "triplicate", 4: "quadruplicate"}.get(n))
    elif mode == "per_group":
        k = rng.choice([2, 3, 4])
        m = rng.choice([3, 4, 5, 6] if is_cells else [4, 5, 6, 8, 10, 12])
        sample_size = k * m
        tpl = pick(rng, REPLICATE_PER_GROUP_ID if is_cells else SUBJECT_PER_GROUP_ID,
                   REPLICATE_PER_GROUP_OOD if is_cells else SUBJECT_PER_GROUP_OOD, ood)
        subj_sent = tpl.format(org=mention, subj=subj, k=k, m=m, k_word=NUM_WORDS[k])
    else:
        sample_size = None
        tpl = pick(rng, REPLICATE_NULL_ID if is_cells else SUBJECT_NULL_ID,
                   REPLICATE_NULL_OOD if is_cells else SUBJECT_NULL_OOD, ood)
        subj_sent = tpl.format(org=mention, cite=f["cite"])
    subj_sent = subj_sent[0].upper() + subj_sent[1:]

    tpl = pick(rng, ASSAY_ID, ASSAY_OOD, ood)
    assay_sent = tpl.format(assay=assay_phrase, Assay=assay_phrase[0].upper() + assay_phrase[1:],
                            cite=f["cite"])

    distractors = [d.format(**f) for d in rng.sample(DISTRACTORS, rng.randint(2, 4))]
    sentences = [subj_sent, assay_sent] + distractors
    rng.shuffle(sentences)
    text = messify(" ".join(sentences), rng)
    return {"text": text,
            "label": {"organism": organism, "assay": assay, "sample_size": sample_size}}


def generate(n, rng, ood, seen):
    out = []
    while len(out) < n:
        ex = make_example(rng, ood)
        if ex["text"] in seen:
            continue
        seen.add(ex["text"])
        out.append(ex)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=300)
    ap.add_argument("--n_val", type=int, default=40)
    ap.add_argument("--n_test", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seen = set()   # texts are unique across all splits
    splits = {
        "train": generate(args.n_train, rng, False, seen),
        "val": generate(args.n_val, rng, False, seen),
        "test_id": generate(args.n_test, rng, False, seen),
        "test_ood": generate(args.n_test, rng, True, seen),
    }
    DATA.mkdir(exist_ok=True)
    for name, rows in splits.items():
        with open(DATA / f"{name}.jsonl", "w") as fh:
            for i, r in enumerate(rows):
                fh.write(json.dumps({"id": f"{name}-{i:04d}", **r}) + "\n")
        print(f"{name:9s} {len(rows):4d} examples -> data/{name}.jsonl")
    print("\nExample (train-0000):\n" + splits["train"][0]["text"])
    print(json.dumps(splits["train"][0]["label"]))


if __name__ == "__main__":
    main()
