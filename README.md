# LASEF — Language-Aware Skeleton Exploration Framework

> **Which Language Should a Skeleton Speak? Language Choices in Multilingual Reasoning**
> Submitted to EMNLP 2026.

Code, data, and results for our study of *skeleton-language choice* in
multilingual reasoning. Skeleton-based prompting is a promising **training-free**
way to structure LLM reasoning, but prior work assumes an English-centric
setting. The **Language-Aware Skeleton Exploration Framework (LASEF)** varies the
*language* of a reasoning skeleton — while fixing its structure and format — to
measure how it affects multilingual math reasoning.

<p align="center">
  <img src="assets/framework_overview.png" width="720" alt="LASEF framework overview"/>
</p>

Anonymous preview during review: <https://anonymous.4open.science/r/LASEF-5EA5/>

---

## TL;DR

A skeleton-based reasoning pass involves **three language choices** for a target
language $\ell_t$:

| Symbol | Meaning |
|--------|---------|
| $\ell_q$ | language of the **query** |
| $\ell_s$ | language of the **skeleton** |
| $\ell_a$ | language of the **reasoning + final answer** |

LASEF fixes the skeleton structure and **only varies** $(\ell_q, \ell_s, \ell_a)$.
Findings: English skeletons give **stable gains** (esp. smaller models /
low-resource languages) but are **not universally optimal**; skeleton-language
effects fall into three patterns (*stable cross-lingual*, *evaluation-dependent*,
*asymmetric*) that **cannot be explained by generation quality alone**.

---

## Repository layout

```
LASEF/
├── README.md  ·  requirements.txt  ·  LICENSE (MIT)
├── LASEF.sh                       # ★ one-shot: evaluate all ℓs, then plot the heatmap
├── scripts/
│   └── src/
│       ├── evaluate.py            # unified inference entry point
│       ├── prompts.py             # shared system prompts & multilingual few-shots
│       ├── translate_skeleton_batch.py      # translation-ablation: MT the skeletons
│       └── postprocess_translated_skeleton.py
├── analysis/
│   └── plot_heatmap.py            # skeleton-language accuracy-Δ heatmap (Fig. 2)
├── data/results/                  # raw model outputs (.jsonl) — see "Large files"
│   ├── MGSM/  MATH500/  PolyMath/             # CoT + skeleton, ± translated query
│   └── MGSM_low_language/                     # low-resource MGSM (34 languages)
│       ├── single_rollout/                   # greedy, per skeleton language ℓs
│       ├── 5-rollout/                        # 5× sampled rollouts
│       └── translated_skeleton_solved/       # translation ablation
├── assets/                        # paper framework figure
└── figure/                        # heatmaps written by plot_heatmap.py
```

`evaluate.py` supersedes the previous family of `eval_combined_*` scripts: one
script, one set of flags.

---

## Quick start

`LASEF.sh` runs the whole pipeline — evaluate every skeleton language, then plot
the heatmap — in one command:

```bash
bash LASEF.sh                                   # defaults: Qwen2.5-7B on MGSM, greedy
ROLLOUT=5 bash LASEF.sh                          # 5 sampled rollouts instead of greedy
TRANSLATE_Q=1 SKELETON_LANGS="en zh ko" bash LASEF.sh   # translated query, subset of ℓs
MODEL=Qwen/Qwen2.5-14B-Instruct DATASET=polymath bash LASEF.sh
```

Configurable via environment variables: `MODEL`, `DATASET`, `LANGS`,
`SKELETON_LANGS`, `ROLLOUT`, `TRANSLATE_Q`, `OUT_DIR`, `FIG`,
`CUDA_VISIBLE_DEVICES`. Results go to `data/results/run/<rollout-tag>/` and the
heatmap to `figure/`. The English skeleton (baseline) is always included so the
delta can be computed. For finer control, call `evaluate.py` directly:

---

## The unified evaluator

Every axis of the framework is a flag on **`scripts/src/evaluate.py`**:

| Flag | Effect |
|------|--------|
| `--method skeleton\|cot` | skeleton-based multi-turn reasoning, or direct CoT |
| `--translate_q` | translate the query into English first ($\ell_q=$ en) |
| `--skeleton_lang en\|zh\|es\|ru\|ko\|th` | skeleton language $\ell_s$ |
| `--decoding greedy\|sampling` | deterministic vs. stochastic decoding |
| `--rollout N` | number of sampled solutions (sampling only) |
| `--translate_cot` | reason in English, then translate the answer back |
| `--existing_skeletons FILE` | reuse pre-generated skeletons; re-solve only |

**Decoding and rollout are decoupled.** `greedy` → temperature 0, one solution;
`sampling` → temperature/top-p from flags, `N` solutions. Skeleton generation
(Turn 1) is always greedy, matching the released data; the decoding flags govern
the solver (answer) generation. The result filename convention is
`{model}-skeleton_multiturn_skelLang-{ℓs}[-Google-transQ][_rolloutN][_exist].jsonl`.

```bash
# Greedy English-skeleton run (→ single_rollout)
python scripts/src/evaluate.py --method skeleton --model Qwen/Qwen2.5-7B-Instruct \
    --dataset mgsm --skeleton_lang en --decoding greedy \
    --output out/Qwen2.5-7B-Instruct-skeleton_multiturn_skelLang-en.jsonl

# 5 rollouts reusing those skeletons (→ 5-rollout)
python scripts/src/evaluate.py --method skeleton --model Qwen/Qwen2.5-7B-Instruct \
    --existing_skeletons out/Qwen2.5-7B-Instruct-skeleton_multiturn_skelLang-en.jsonl \
    --decoding sampling --rollout 5 \
    --output out/Qwen2.5-7B-Instruct-skeleton_multiturn_skelLang-en_rollout5_exist.jsonl

# Translated-query CoT baseline
python scripts/src/evaluate.py --method cot --model Qwen/Qwen2.5-7B-Instruct \
    --dataset mgsm --translate_q --output out/cot-transQ.jsonl
```

Datasets (`--dataset`) are local benchmark keys, keyword-matched to the JSON
files in `data/test_data/`: **`mgsm`**, **`mgsm_low`** (34 low-resource
languages), **`math500`**, **`polymath`**.

`LASEF.sh` (see [Quick start](#quick-start)) chains these into a full sweep and
plots the heatmap in one command.

---

## Analysis: skeleton-language heatmap

`analysis/plot_heatmap.py` reads a directory of skeleton results and renders the
Exact-Match accuracy **delta of each skeleton language $\ell_s$ vs. the English
skeleton**, per answer language $\ell_a$. Correctness uses `math_verify`; fastText
`lid.176` enforces the $\ell_a = \ell_t$ validity constraint and verifies each
skeleton was produced in the intended language. Rows (answer languages) and
columns (skeleton languages) are auto-detected from the data.

```bash
python analysis/plot_heatmap.py \
    --data_dir data/results/MGSM_low_language/single_rollout \
    --model Qwen2.5-7B-Instruct \
    --output figure/single_rollout_heatmap.pdf
```

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.12 recommended
pip install -r requirements.txt

# fastText language-ID model (required by plot_heatmap.py); not shipped here
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
export LASEF_ROOT="$(pwd)"      # scripts resolve paths from here (or auto-detect)
```

Benchmarks are read from **local JSON files** under `data/test_data/`
(`MGSM-translated.json`, `MGSM-low-resource-translated.json`,
`MATH-500-translated.json`, `PolyMath-translated.json`) — no external dataset hub
is contacted, keeping released runs self-contained and anonymous. Each file is a
flat list of rows carrying every language; `evaluate.py` filters by language (and
PolyMath difficulty) and uses the English column for the `en` target. Inference
uses vLLM and assumes a multi-GPU node; the released `data/results/` lets you
reproduce the heatmaps **without GPUs**.

**Models**: Qwen2.5-Instruct (7B/14B/32B/72B), Llama-3.1-Instruct (8B/70B).
**Benchmarks**: MGSM, MATH-500, PolyMath. **Skeleton languages** $\ell_s$:
en, zh, es, ru, ko, th. **Metric**: Exact Match, restricted to $\ell_a=\ell_t$.

---

## Data

Input benchmarks live in `data/test_data/` (tracked) and are loaded locally — no
external dataset hub is contacted.

**Raw model outputs (`data/results/`) are not in this repository.** They are
several GB with individual files exceeding GitHub's 100 MB per-file limit. To
obtain them, either re-run inference with `LASEF.sh`, or download the released
archive (see the [anonymous mirror](https://anonymous.4open.science/r/LASEF-5EA5/))
and unpack it into `data/results/`. The processed heatmaps under `figure/` are
also regenerated locally via `analysis/plot_heatmap.py`.


Released under the [MIT License](LICENSE).
