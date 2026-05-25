# LASEF — Language-Aware Skeleton Exploration Framework

> **Which Language Should a Skeleton Speak? Language Choices in Multilingual Reasoning**

Code and input data for our study of *skeleton-language choice* in multilingual
reasoning. Skeleton-based prompting is a promising **training-free** way to
structure LLM reasoning, but prior work assumes an English-centric setting. The
**Language-Aware Skeleton Exploration Framework (LASEF)** varies the *language*
of a reasoning skeleton — while fixing its structure and format — to measure how
it affects multilingual math reasoning.

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
Findings: English skeletons give **stable gains** (especially for smaller models
and low-resource languages) but are **not universally optimal**; skeleton-language
effects fall into three patterns (*stable cross-lingual*, *evaluation-dependent*,
*asymmetric*) that **cannot be explained by generation quality alone**.

---

## Repository layout

```
LASEF/
├── LASEF.sh                       # ★ one-shot: evaluate every ℓs, then plot the heatmap
├── scripts/src/
│   ├── evaluate.py                # unified inference entry point (CoT / skeleton)
│   ├── prompts.py                 # shared system prompts & multilingual few-shots
│   ├── translate_skeleton_batch.py        # translation ablation: machine-translate skeletons
│   └── postprocess_translated_skeleton.py
├── analysis/
│   └── plot_heatmap.py            # skeleton-language accuracy-Δ heatmap
├── data/test_data/                # input benchmarks (local JSON; loaded directly)
│   ├── MGSM-translated.json
│   ├── MGSM-low-resource-translated.json
│   ├── MATH-500-translated.json
│   └── PolyMath-translated.json
├── figure/                        # heatmaps written by plot_heatmap.py (git-ignored)
├── requirements.txt  ·  LICENSE (MIT)  ·  README.md
```

> **Note.** Raw model outputs are **not** in this repository (multiple GB, with
> files over GitHub's 100 MB limit). Generate them with `LASEF.sh`; results are
> written under `data/results/`, which is git-ignored. See [Data](#data).

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.12 recommended
pip install -r requirements.txt

# fastText language-ID model (required by plot_heatmap.py); not shipped here
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
export LASEF_ROOT="$(pwd)"      # scripts resolve paths from here (or auto-detect)
```

Inference uses vLLM and assumes a multi-GPU node. **Models**: Qwen2.5-Instruct
(7B/14B/32B/72B), Llama-3.1-Instruct (8B/70B). **Benchmarks**: MGSM, MATH-500,
PolyMath. **Skeleton languages** $\ell_s$: en, zh, es, ru, ko, th.
**Metric**: Exact Match, restricted to $\ell_a = \ell_t$ (the answer must be in
the target language), with fastText `lid.176` for language identification.

---

## Quick start

`LASEF.sh` runs the whole pipeline — evaluate every skeleton language, then plot
the heatmap — in one command:

```bash
bash LASEF.sh                                            # Qwen2.5-7B on MGSM, greedy
ROLLOUT=5 bash LASEF.sh                                  # 5 sampled rollouts instead of greedy
TRANSLATE_Q=1 SKELETON_LANGS="en zh ko" bash LASEF.sh    # translated query, subset of ℓs
MODEL=Qwen/Qwen2.5-14B-Instruct DATASET=polymath bash LASEF.sh
```

Configurable via environment variables: `MODEL`, `DATASET`, `LANGS`,
`SKELETON_LANGS`, `ROLLOUT`, `TRANSLATE_Q`, `OUT_DIR`, `FIG`,
`CUDA_VISIBLE_DEVICES`. Per-skeleton-language results are written to
`data/results/run/<rollout-tag>/` and the heatmap to `figure/`. The English
skeleton (baseline) is always included so the delta can be computed.

---

## The unified evaluator

`evaluate.py` is a single entry point; every axis of the framework is a flag:

| Flag | Effect |
|------|--------|
| `--method skeleton\|cot` | skeleton-based multi-turn reasoning, or direct CoT |
| `--dataset mgsm\|mgsm_low\|math500\|polymath` | local benchmark (keyword-matched) |
| `--translate_q` | translate the query into English first ($\ell_q =$ en) |
| `--skeleton_lang en\|zh\|es\|ru\|ko\|th` | skeleton language $\ell_s$ |
| `--decoding greedy\|sampling` | deterministic vs. stochastic decoding |
| `--rollout N` | number of sampled solutions (sampling only) |
| `--translate_cot` | reason in English, then translate the answer back |
| `--existing_skeletons FILE` | reuse pre-generated skeletons; re-solve only |

**Decoding and rollout are decoupled.** `greedy` → temperature 0, one solution;
`sampling` → temperature/top-p from flags, `N` solutions. Skeleton generation
(Turn 1) is always greedy; the decoding flags govern the solver (answer)
generation. The output filename convention is
`{model}-skeleton_multiturn_skelLang-{ℓs}.jsonl` (one file per skeleton language,
placed in a rollout-tagged folder such as `single_rollout/` or `5-rollout/`).

```bash
# Greedy English-skeleton run
python scripts/src/evaluate.py --method skeleton --model Qwen/Qwen2.5-7B-Instruct \
    --dataset mgsm --skeleton_lang en --decoding greedy \
    --output runs/single_rollout/Qwen2.5-7B-Instruct-skeleton_multiturn_skelLang-en.jsonl

# 5 rollouts reusing those skeletons
python scripts/src/evaluate.py --method skeleton --model Qwen/Qwen2.5-7B-Instruct \
    --existing_skeletons runs/single_rollout/Qwen2.5-7B-Instruct-skeleton_multiturn_skelLang-en.jsonl \
    --decoding sampling --rollout 5 \
    --output runs/5-rollout/Qwen2.5-7B-Instruct-skeleton_multiturn_skelLang-en.jsonl

# Translated-query CoT baseline
python scripts/src/evaluate.py --method cot --model Qwen/Qwen2.5-7B-Instruct \
    --dataset mgsm --translate_q --output runs/cot-transQ.jsonl
```

---

## Analysis: skeleton-language heatmap

`analysis/plot_heatmap.py` reads a directory of skeleton results and renders the
Exact-Match accuracy **delta of each skeleton language $\ell_s$ vs. the English
skeleton**, per answer language $\ell_a$. Correctness uses `math_verify`; fastText
`lid.176` enforces the $\ell_a = \ell_t$ constraint and verifies each skeleton was
produced in the intended language. Rows (answer languages) and columns (skeleton
languages) are auto-detected from the data, and the rollout count is inferred from
the folder name (`single_rollout`, `5-rollout`, …).

```bash
python analysis/plot_heatmap.py \
    --data_dir data/results/run/single_rollout \
    --model Qwen2.5-7B-Instruct \
    --output figure/single_rollout_heatmap.pdf
```

---

## Data

Input benchmarks live in `data/test_data/` (tracked) and are loaded **locally** —
no external dataset hub is contacted, keeping runs self-contained and anonymous.
Each file is a flat JSON list of rows carrying every language; `evaluate.py`
filters by language (and, for PolyMath, difficulty) and uses the English column
for the `en` target.

| `--dataset` | File | Coverage |
|-------------|------|----------|
| `mgsm`      | `MGSM-translated.json`              | MGSM, high-resource languages |
| `mgsm_low`  | `MGSM-low-resource-translated.json` | MGSM, 34 low-resource languages |
| `math500`   | `MATH-500-translated.json`          | MATH-500 |
| `polymath`  | `PolyMath-translated.json`          | PolyMath (top/high/medium/low) |

**Raw model outputs are not included.** Reproduce them by running `LASEF.sh`
(written to the git-ignored `data/results/`), then plot with `plot_heatmap.py`.

---

Released under the [MIT License](LICENSE).
