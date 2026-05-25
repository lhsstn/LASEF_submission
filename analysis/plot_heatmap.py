# -*- coding: utf-8 -*-
"""Skeleton-language performance heatmap (LASEF Fig. 2).

For a directory of skeleton results produced by ``scripts/src/evaluate.py``, this
computes, for every (skeleton language ell_s, answer language ell_a) pair, the
Exact-Match accuracy delta of that skeleton language relative to the **English
skeleton** baseline, and renders it as a heatmap.

Accuracy uses ``math_verify`` for answer correctness and fastText ``lid.176`` for
language identification: a rollout counts only if the response is actually in the
target answer language (the ``ell_a = ell_t`` validity constraint from the paper),
and a sample is included only if its skeleton was generated in the intended
language ell_s.

Usage
-----
    python analysis/plot_heatmap.py \
        --data_dir data/results/SkelLang_MGSM/single_rollout \
        --model Qwen2.5-7B-Instruct \
        --output figure/single_rollout/heatmap.pdf

File naming is auto-detected; ``--translated`` / ``--rollout_count`` are inferred
from the directory name unless given explicitly.
"""

import argparse
import json
import os
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Skeleton languages compared against the English-skeleton baseline (columns).
# Defaults; overridden by auto-detection / CLI in main().
SKELETON_LANGUAGES = ["zh", "es", "ru", "ko", "th"]
# Answer / query languages evaluated (rows). Auto-detected from the data by default.
ANSWER_LANGUAGES = ["en", "zh", "es", "ko", "th", "sw", "te"]
LANGUAGE_NAMES = {
    "en": "English", "zh": "Chinese", "es": "Spanish", "ru": "Russian",
    "ko": "Korean", "th": "Thai", "sw": "Swahili", "te": "Telugu", "bn": "Bengali",
    "ta": "Tamil", "kn": "Kannada", "my": "Burmese", "km": "Khmer", "am": "Amharic",
    "yo": "Yoruba", "si": "Sinhala", "gu": "Gujarati", "ne": "Nepali", "uz": "Uzbek",
    "ky": "Kyrgyz", "ceb": "Cebuano", "eu": "Basque", "gn": "Guarani", "hy": "Armenian",
    "jv": "Javanese", "ka": "Georgian", "kk": "Kazakh", "ku": "Kurdish", "lo": "Lao",
    "mg": "Malagasy", "ml": "Malayalam", "mn": "Mongolian", "mr": "Marathi",
    "mt": "Maltese", "or": "Odia", "pa": "Punjabi", "ps": "Pashto", "qu": "Quechua",
    "sd": "Sindhi", "so": "Somali", "su": "Sundanese", "tg": "Tajik", "ug": "Uyghur",
}

_LANG_MODEL = None  # lazily loaded fastText model (per process)
_FASTTEXT_PATH = None
ROLLOUT_COUNT = 1   # set in main()


# ─────────────────────────────────────────────────────────────────────────────
# Language identification & correctness
# ─────────────────────────────────────────────────────────────────────────────

def _get_lang_model():
    global _LANG_MODEL
    if _LANG_MODEL is None:
        import fasttext
        fasttext.FastText.eprint = lambda x: None
        _LANG_MODEL = fasttext.load_model(_FASTTEXT_PATH)
    return _LANG_MODEL


def detect_language(text):
    if not isinstance(text, str):
        return "unk"
    text = text.replace("\n", " ").strip()
    if len(text) < 2:
        return "unk"
    try:
        labels, probs = _get_lang_model().predict(text, k=1)
        return labels[0].replace("__label__", "") if probs is not None else "unk"
    except Exception:
        return "unk"


def load_jsonl(path):
    if not os.path.exists(path):
        return None
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except Exception:
                pass
    return data


def add_detected_languages(data, detect_skeleton=False):
    for item in tqdm(data, desc="lang-id", leave=False):
        responses = item.get("responses", [])[:ROLLOUT_COUNT]
        item["detected_response_languages"] = [
            detect_language(r if isinstance(r, str) else "") for r in responses]
        if detect_skeleton:
            sk = item.get("skeleton", [])
            sk_text = sk[0] if isinstance(sk, list) and sk else (sk if isinstance(sk, str) else "")
            item["detected_skeleton_language"] = detect_language(sk_text)
    return data


def group_by_language(data):
    grouped = defaultdict(list)
    for item in data:
        grouped[item.get("question_language", "unknown")].append(item)
    return grouped


def _item_id(item):
    oid = item.get("original_id")
    return str(oid if oid is not None else item.get("global_id"))


def evaluate_pair(args):
    """Strict (language-compliant) accuracy of baseline vs. target for one sample."""
    item_a, item_b, answer_lang = args
    resp_a = item_a.get("responses", [])[:ROLLOUT_COUNT]
    resp_b = item_b.get("responses", [])[:ROLLOUT_COUNT]
    langs_a = item_a.get("detected_response_languages", [])[:ROLLOUT_COUNT]
    langs_b = item_b.get("detected_response_languages", [])[:ROLLOUT_COUNT]
    try:
        from math_verify import parse, verify
        gold = parse(str(item_a["answer"]))
        strict_a, strict_b = [], []
        for r in range(min(len(resp_a), len(resp_b))):
            comp_a = langs_a[r] == answer_lang
            comp_b = langs_b[r] == answer_lang
            try:
                ca = int(verify(gold, parse(str(resp_a[r]))))
            except Exception:
                ca = 0
            try:
                cb = int(verify(gold, parse(str(resp_b[r]))))
            except Exception:
                cb = 0
            strict_a.append(ca if comp_a else 0)
            strict_b.append(cb if comp_b else 0)
        avg_a = sum(strict_a) / len(strict_a) if strict_a else 0.0
        avg_b = sum(strict_b) / len(strict_b) if strict_b else 0.0
        return avg_a, avg_b, 0
    except Exception:
        return 0.0, 0.0, 1


# ─────────────────────────────────────────────────────────────────────────────
# File resolution (mirrors evaluate.py output naming)
# ─────────────────────────────────────────────────────────────────────────────

def resolve_path(data_dir, model, skel_lang, translated, rollout, rollout_count, exist):
    direct = os.path.join(data_dir, f"{model}-skeleton_multiturn_skelLang-{skel_lang}.jsonl")
    if os.path.exists(direct):
        return direct
    suffix = f"_rollout{rollout_count}" if rollout else ""
    if translated:
        return os.path.join(data_dir, f"{model}-skeleton_multiturn_skelLang-{skel_lang}-Google-transQ{suffix}.jsonl")
    if exist:
        return os.path.join(data_dir, f"{model}-skeleton_multiturn_skelLang-{skel_lang}{suffix}_exist.jsonl")
    return direct


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Plot skeleton-language accuracy-delta heatmap.")
    p.add_argument("--data_dir", required=True, help="Directory of skeleton result JSONLs")
    p.add_argument("--model", default="Qwen2.5-7B-Instruct")
    p.add_argument("--output", default=None, help="Output PDF path (default: <data_dir>/heatmap.pdf)")
    p.add_argument("--fasttext_model", default=None,
                   help="Path to lid.176.bin (default: $LASEF_ROOT/lid.176.bin)")
    p.add_argument("--translated", choices=["auto", "true", "false"], default="auto")
    p.add_argument("--rollout_count", default="auto")
    p.add_argument("--skeleton_langs", default="auto",
                   help="Comma-separated ℓs columns, or 'auto' to detect skelLang-* files")
    p.add_argument("--answer_langs", default="auto",
                   help="Comma-separated ℓa rows, or 'auto' to detect from the baseline data")
    p.add_argument("--min_samples", type=int, default=10,
                   help="Skip a cell with fewer than this many valid aligned samples")
    return p.parse_args()


def detect_skeleton_langs(data_dir, model):
    """Skeleton languages with a present skelLang-*.jsonl file, excluding the 'en' baseline."""
    import glob
    found = []
    prefix = f"{model}-skeleton_multiturn_skelLang-"
    for path in sorted(glob.glob(os.path.join(data_dir, f"{prefix}*.jsonl"))):
        name = os.path.basename(path)[len(prefix):]
        code = name.split("-")[0].split("_")[0].replace(".jsonl", "")
        if code and code != "en" and code not in found:
            found.append(code)
    return found


def main():
    global _FASTTEXT_PATH, ROLLOUT_COUNT, SKELETON_LANGUAGES, ANSWER_LANGUAGES
    args = parse_args()

    root = os.environ.get("LASEF_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _FASTTEXT_PATH = args.fasttext_model or os.path.join(root, "lid.176.bin")
    if not os.path.exists(_FASTTEXT_PATH):
        raise SystemExit(f"fastText model not found: {_FASTTEXT_PATH}\n"
                         "Download lid.176.bin (see README) or pass --fasttext_model.")

    # Normalize folder name so '5-rollout' and '5_rollout' are treated the same.
    folder = os.path.basename(os.path.normpath(args.data_dir)).replace("-", "_").lower()
    if args.rollout_count == "auto":
        ROLLOUT_COUNT = next((n for tag, n in
                              [("3_rollout", 3), ("5_rollout", 5), ("10_rollout", 10),
                               ("single_rollout", 1)] if tag in folder), 5)
    else:
        ROLLOUT_COUNT = int(args.rollout_count)
    rollout = "single" not in folder and ROLLOUT_COUNT > 1
    exist = "exist" in folder
    if args.translated == "auto":
        translated = "transq" in folder or "translated" in folder
    else:
        translated = args.translated == "true"

    # Resolve skeleton-language columns.
    if args.skeleton_langs != "auto":
        SKELETON_LANGUAGES = [s.strip() for s in args.skeleton_langs.split(",") if s.strip()]
    else:
        detected = detect_skeleton_langs(args.data_dir, args.model)
        if detected:
            SKELETON_LANGUAGES = detected
    print(f"🧭 Skeleton languages (ℓs): {SKELETON_LANGUAGES}")

    out_path = args.output or os.path.join(args.data_dir, f"{args.model}_skeleton_delta_heatmap.pdf")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    print(f"📂 data_dir={args.data_dir} | rollout_count={ROLLOUT_COUNT} | "
          f"translated={translated} | exist={exist}")

    # Baseline: English skeleton.
    base_path = resolve_path(args.data_dir, args.model, "en", translated, rollout, ROLLOUT_COUNT, exist)
    baseline = load_jsonl(base_path)
    if not baseline:
        raise SystemExit(f"❌ Baseline (English skeleton) not found: {base_path}")
    print(f"✅ Baseline: {os.path.basename(base_path)} ({len(baseline)} items)")
    baseline = add_detected_languages(baseline, detect_skeleton=False)
    baseline_grouped = group_by_language(baseline)

    # Resolve answer-language rows (auto = whatever query languages appear in the data).
    if args.answer_langs != "auto":
        ANSWER_LANGUAGES = [s.strip() for s in args.answer_langs.split(",") if s.strip()]
    else:
        ANSWER_LANGUAGES = sorted(k for k in baseline_grouped if k not in (None, "unknown"))
    print(f"🧭 Answer languages (ℓa): {len(ANSWER_LANGUAGES)} -> {ANSWER_LANGUAGES}")

    delta_matrix = defaultdict(dict)
    for skel_lang in SKELETON_LANGUAGES:
        path = resolve_path(args.data_dir, args.model, skel_lang, translated, rollout, ROLLOUT_COUNT, exist)
        target = load_jsonl(path)
        if not target:
            print(f"   ⚠️ {skel_lang}: file missing, skipping ({os.path.basename(path)})")
            continue
        target = add_detected_languages(target, detect_skeleton=True)
        target_grouped = group_by_language(target)
        print(f"📊 skeleton={skel_lang} ({len(target)} items)")

        for answer_lang in ANSWER_LANGUAGES:
            base_l = baseline_grouped.get(answer_lang, [])
            tgt_l = target_grouped.get(answer_lang, [])
            if not base_l or not tgt_l:
                continue
            dict_b = {_item_id(it): it for it in tgt_l}
            pairs = []
            for it_a in base_l:
                it_b = dict_b.get(_item_id(it_a))
                if it_b is None:
                    continue
                la = it_a.get("detected_response_languages", [])
                lb = it_b.get("detected_response_languages", [])
                if answer_lang not in la or answer_lang not in lb:
                    continue
                if it_b.get("detected_skeleton_language", "unk") != skel_lang:
                    continue
                pairs.append((it_a, it_b))
            if len(pairs) < args.min_samples:
                continue
            with Pool(cpu_count()) as pool:
                results = pool.map(evaluate_pair, [(a, b, answer_lang) for a, b in pairs])
            valid = [r for r in results if r[2] == 0]
            if not valid:
                continue
            acc_a = sum(r[0] for r in valid) / len(valid) * 100
            acc_b = sum(r[1] for r in valid) / len(valid) * 100
            delta_matrix[skel_lang][answer_lang] = acc_b - acc_a
            print(f"   [{answer_lang}] {acc_a:.1f}->{acc_b:.1f} (Δ{acc_b - acc_a:+.1f}) N={len(valid)}")

    # Rows = answer language, cols = skeleton language.
    df = pd.DataFrame(index=ANSWER_LANGUAGES, columns=SKELETON_LANGUAGES, dtype=float)
    for skel_lang in SKELETON_LANGUAGES:
        for answer_lang in ANSWER_LANGUAGES:
            df.loc[answer_lang, skel_lang] = delta_matrix.get(skel_lang, {}).get(answer_lang, np.nan)
    df.loc["AVG"] = df.mean(axis=0, skipna=True)

    # Height scales with the number of answer-language rows (34 for low-resource).
    plt.figure(figsize=(max(6, 1.2 * len(SKELETON_LANGUAGES) + 2), max(5, 0.4 * len(df) + 1)))
    ax = sns.heatmap(
        df.astype(float), annot=True, fmt=".1f", cmap="RdBu_r", center=0,
        vmin=-15, vmax=15, linewidths=0.5,
        xticklabels=[f"{l.upper()}\n({LANGUAGE_NAMES.get(l, l)})" for l in SKELETON_LANGUAGES],
        yticklabels=[*[l.upper() for l in ANSWER_LANGUAGES], "AVG"],
        cbar_kws={"label": "Exact-Match Δ vs. English skeleton (%)"},
    )
    avg_idx = df.index.get_loc("AVG")
    ax.hlines([avg_idx, avg_idx + 1], xmin=0, xmax=df.shape[1], colors="black", linewidth=2.0)
    for tick in ax.get_yticklabels():
        if tick.get_text() == "AVG":
            tick.set_fontweight("bold")
    plt.title(f"Skeleton-Language Accuracy Δ\n{args.model}\n(skeleton ℓs − English skeleton)")
    plt.xlabel("Skeleton language (ℓs)")
    plt.ylabel("Answer language (ℓa)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ Heatmap saved to: {out_path}")


if __name__ == "__main__":
    main()
