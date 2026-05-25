# -*- coding: utf-8 -*-
"""Unified LASEF evaluation entry point.

A single script that supersedes the previous family of `eval_combined_*` files.
It runs Chain-of-Thought or skeleton-based reasoning with vLLM and exposes every
axis of the Language-Aware Skeleton Exploration Framework through flags:

  * Translate-query (ell_q)   --translate_q          query translated to English first
  * Skeleton language (ell_s) --skeleton_lang en|zh|es|ru|ko|th
  * Decoding                  --decoding greedy|sampling
  * Rollouts                  --rollout N            number of sampled solutions
  * Reuse skeletons           --existing_skeletons FILE   skip Turn-1, re-solve only

Decoding vs. rollout are decoupled:
  greedy   -> temperature 0, top_p 1, one solution (rollout forced to 1)
  sampling -> temperature/top_p from flags, `rollout` solutions per item

Skeleton generation (Turn 1) is always greedy, matching the released data; the
`--decoding` / `--rollout` knobs control the solver (answer) generation only.

Datasets are loaded from local JSON files under ``data/test_data`` (no external
dataset hub) so that released runs stay fully self-contained and anonymous.

Examples
--------
# Greedy, English skeleton, native query  (single_rollout)
python evaluate.py --method skeleton --model Qwen/Qwen2.5-7B-Instruct \
    --dataset mgsm --output out.jsonl --skeleton_lang en

# Sampling, 10 rollouts, reuse skeletons from a greedy run  (10_rollout)
python evaluate.py --method skeleton --model Qwen/Qwen2.5-7B-Instruct \
    --output out_r10.jsonl --decoding sampling --rollout 10 \
    --existing_skeletons single_rollout/...skelLang-en.jsonl

# Translated-query CoT baseline
python evaluate.py --method cot --model Qwen/Qwen2.5-7B-Instruct \
    --dataset mgsm --output cot_transQ.jsonl --translate_q
"""

import argparse
import json
import os
import random

from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

import prompts

# Token only for downloading public model weights (e.g. gated Llama); never used
# for datasets, which are read locally from data/test_data.
HF_TOKEN = os.getenv("HF_TOKEN")


# ═══════════════════════════════════════════════════════════════════════════════
#                           DATASET CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════════

# Loading semantics per benchmark type (drives build_load_targets / sampling).
DATASET_CONFIGS = {
    "polymath": {"type": "polymath"},
    "math500": {"type": "math500"},
    "mgsm": {"type": "mgsm"},
    "default": {"type": "default"},
}

# Local datasets under data/test_data. Each is a flat JSON list of rows; we filter
# by language (and, for PolyMath, difficulty) and normalize to {question, answer,
# id, difficulty}. For target language "en" we use the English column.
#   lang_col   : column holding the language code of each row
#   q_col      : column with the (target-language) question text
#   en_col     : column with the English question text (used when lang == "en")
#   diff_col   : column with the difficulty split (PolyMath); else None
#   diff_field : column to copy into the output "difficulty" field (e.g. MATH level)
LOCAL_SPECS = {
    "mgsm":     {"file": "MGSM-translated.json",              "lang_col": "lang",
                 "q_col": "question",           "en_col": "translated_en",
                 "ans_col": "answer", "id_col": None,        "diff_col": None, "diff_field": None},
    "mgsm_low": {"file": "MGSM-low-resource-translated.json", "lang_col": "language",
                 "q_col": "translate_question", "en_col": "en_question",
                 "ans_col": "answer", "id_col": "id",        "diff_col": None, "diff_field": None},
    "math500":  {"file": "MATH-500-translated.json",          "lang_col": "split",
                 "q_col": "problem",            "en_col": "translated_en",
                 "ans_col": "answer", "id_col": "unique_id", "diff_col": None, "diff_field": "level"},
    "polymath": {"file": "PolyMath-translated.json",          "lang_col": "lang",
                 "q_col": "question",           "en_col": "translated_en",
                 "ans_col": "answer", "id_col": "id",        "diff_col": "split", "diff_field": None},
}


def get_dataset_config(dataset_name: str) -> dict:
    name = dataset_name.lower()
    if "polymath" in name or "poly" in name: return DATASET_CONFIGS["polymath"]
    if "math-500" in name or "math500" in name or "math" in name: return DATASET_CONFIGS["math500"]
    if "mgsm" in name or "gsm" in name: return DATASET_CONFIGS["mgsm"]
    return DATASET_CONFIGS["default"]


def resolve_local_spec_key(dataset_name: str) -> str:
    """Map a dataset id/keyword to a LOCAL_SPECS key."""
    name = dataset_name.lower()
    if "low" in name: return "mgsm_low"
    if "poly" in name: return "polymath"
    if "math" in name: return "math500"
    if "mgsm" in name or "gsm" in name: return "mgsm"
    raise SystemExit(f"Cannot map dataset '{dataset_name}' to a local file in data/test_data. "
                     f"Use one of: mgsm, mgsm_low, math500, polymath.")


def load_local_file(data_root: str, spec: dict) -> list:
    path = os.path.join(data_root, spec["file"])
    if not os.path.exists(path):
        raise SystemExit(f"Local dataset not found: {path}\n"
                         f"Place the benchmark JSON files under {data_root}.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _normalize_row(row: dict, spec: dict, difficulty, use_en: bool, idx: int) -> dict:
    question = row.get(spec["en_col"]) if use_en else row.get(spec["q_col"], "")
    rid = row.get(spec["id_col"]) if spec["id_col"] else idx
    if spec["diff_field"]:
        diff = row.get(spec["diff_field"], difficulty)
    elif spec["diff_col"]:
        diff = row.get(spec["diff_col"], difficulty)
    else:
        diff = difficulty
    return {"question": question, "answer": row.get(spec["ans_col"]),
            "id": rid, "difficulty": diff}


def select_local_rows(rows: list, spec: dict, lang: str, difficulty) -> list:
    """Filter the flat row list for one (language, difficulty) and normalize it.

    English is not stored as its own language in these translated benchmarks, so
    we reconstruct a single canonical English set from one reference language
    group, reading the English column. (Per-row English text / IDs are not shared
    across languages, so naive de-duplication would over-count.)
    """
    def matches_difficulty(r):
        return not (difficulty is not None and spec["diff_col"]
                    and str(r.get(spec["diff_col"])) != str(difficulty))

    out = []
    if lang == "en":
        avail = []
        for r in rows:
            lv = str(r.get(spec["lang_col"]))
            if lv not in avail and lv not in ("en", "None"):
                avail.append(lv)
        ref_lang = sorted(avail)[0] if avail else None
        for r in rows:
            if ref_lang is not None and str(r.get(spec["lang_col"])) != ref_lang:
                continue
            if not matches_difficulty(r):
                continue
            out.append(_normalize_row(r, spec, difficulty, use_en=True, idx=len(out)))
    else:
        for r in rows:
            if str(r.get(spec["lang_col"])) != lang:
                continue
            if not matches_difficulty(r):
                continue
            out.append(_normalize_row(r, spec, difficulty, use_en=False, idx=len(out)))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#                              ARGUMENT PARSER
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Unified LASEF evaluation (CoT / skeleton, translate-query, "
                    "skeleton language, greedy/sampling, rollouts)."
    )
    # What to run
    p.add_argument("--method", choices=["skeleton", "cot"], default="skeleton",
                   help="skeleton = skeleton-based multi-turn reasoning; cot = direct CoT")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset",
                   help="Local benchmark: mgsm | mgsm_low | math500 | polymath "
                        "(keyword-matched; required unless --existing_skeletons)")
    p.add_argument("--data_root", default=None,
                   help="Directory of local benchmark JSON files (default: $LASEF_ROOT/data/test_data)")
    p.add_argument("--output", required=True)
    p.add_argument("--lora", default=None)

    # LASEF language axes
    p.add_argument("--translate_q", action="store_true",
                   help="Translate the query into English before reasoning (ell_q = en)")
    p.add_argument("--translate_cot", action="store_true",
                   help="Reason/answer in English, then translate back to the target language")
    p.add_argument("--skeleton_lang", default="en",
                   help="Skeleton language ell_s (en, zh, es, ru, ko, th)")

    # Decoding (decoupled from rollout)
    p.add_argument("--decoding", choices=["greedy", "sampling"], default="greedy",
                   help="greedy: temp 0, single solution; sampling: stochastic, `rollout` solutions")
    p.add_argument("--rollout", type=int, default=1,
                   help="Number of sampled solutions per item (only used when --decoding sampling)")
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    p.add_argument("--top_p", type=float, default=0.95, help="Sampling top-p")
    p.add_argument("--top_k", type=float, default=-1, help="Sampling top-k (-1 = disabled)")

    # Reuse pre-generated skeletons (skip Turn 1)
    p.add_argument("--existing_skeletons", default=None,
                   help="JSONL with a 'skeleton' field per item; reuse instead of generating")

    # Inference / infra
    p.add_argument("--tp", type=int, default=8, help="Tensor-parallel size")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--gpu_mem", type=float, default=0.9)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--reasoning_model", action="store_true",
                   help="Model uses an explicit <think> block (CoT only)")

    # Task scope
    p.add_argument("--langs", default="en,es,zh,ko,th,sw,te",
                   help="Comma-separated target/query languages")
    p.add_argument("--difficulties", default="top,high,medium,low",
                   help="PolyMath difficulty splits")
    p.add_argument("--modes", default="all", help="CoT modes (see normalize_modes)")
    p.add_argument("--sample_ratio", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
#                              DATA HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def sample_rows(rows: list, ratio: float, seed: int) -> list:
    if ratio >= 1.0:
        return rows
    k = max(1, int(len(rows) * ratio))
    random.seed(seed)
    return [rows[i] for i in sorted(random.sample(range(len(rows)), k))]


def get_qa_text(row: dict, lang: str) -> str:
    col = f"question_{lang}"
    if row.get(col):
        return row[col]
    return row.get("question", "")


def chunks(iterable, size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_load_targets(config: dict, langs, diffs):
    targets = []
    t = config["type"]
    if t == "polymath":
        for l in langs:
            for d in diffs:
                targets.append({"lang": l, "split": d, "name": l})
    elif t in ("math500", "aime25"):
        for l in langs:
            targets.append({"lang": l, "split": l, "name": None})
    elif t == "mgsm":
        for l in langs:
            targets.append({"lang": l, "split": "test", "name": l})
    else:
        for d in diffs:
            targets.append({"lang": "en", "split": d, "name": None})
    return targets


# ═══════════════════════════════════════════════════════════════════════════════
#                              PROMPT BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_translation_prompt(tok, text: str, src_lang: str, tgt_lang: str) -> str:
    fewshots = prompts.TRANSLATION_FEWSHOTS.get(f"{src_lang}_to_{tgt_lang}", [])
    src_name, tgt_name = prompts.get_lang_name(src_lang), prompts.get_lang_name(tgt_lang)
    messages = [{"role": "system", "content": prompts.SYSTEM_PROMPT_TRANSLATOR}]
    for shot in fewshots[:5]:
        messages.append({"role": "user",
                         "content": f"Translate from {src_name} to {tgt_name}:\n{shot['src']}"})
        messages.append({"role": "assistant", "content": shot["tgt"]})
    messages.append({"role": "user",
                     "content": f"Translate from {src_name} to {tgt_name}:\n{text}"})
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_skeleton_prompt(tok, query_lang: str, question: str, skeleton_lang: str) -> str:
    """Turn 1: generate a skeleton in `skeleton_lang` (ell_s) for a query in `query_lang`."""
    questions = prompts.SKELETON_QUESTION_FEWSHOTS.get(
        query_lang, prompts.SKELETON_QUESTION_FEWSHOTS["en"])
    answers = prompts.SKELETON_ANSWER_FEWSHOTS.get(
        skeleton_lang, prompts.SKELETON_ANSWER_FEWSHOTS["en"])
    system = prompts.SYSTEM_PROMPT_SKELETON.format(target_lang=prompts.get_lang_name(skeleton_lang))

    messages = [{"role": "system", "content": system}]
    for i, q in enumerate(questions):
        messages.append({"role": "user", "content": q})
        if i < len(answers):
            messages.append({"role": "assistant", "content": answers[i]})
    messages.append({"role": "user",
                     "content": f"Question: {question}\n\n"
                                f"You must respond in {prompts.get_lang_name(skeleton_lang)}."})
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_solver_prompt(tok, question: str, skeleton: str, answer_lang: str,
                        force_english: bool = False) -> str:
    """Turn 2: solve, following the skeleton, answering in `answer_lang` (ell_a)."""
    reason_lang = "en" if force_english else answer_lang
    lang_name = prompts.get_lang_name(reason_lang)
    lang_hint = (
        "Instructions:\n"
        "1. Follow the Reasoning Skeleton above.\n"
        "2. Verify numbers against the Question text.\n"
        f"3. Let's think step-by-step. Respond in {lang_name}.\n"
        "Final Answer Format: \\boxed{answer}"
    )
    user_prompt = f"Question: {question}\n\nReasoning Skeleton:\n{skeleton}\n\n{lang_hint}"
    messages = [{"role": "user", "content": user_prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text + prompts.SOLVER_TRIGGERS.get(reason_lang, "")


def build_cot_prompt(tok, prompt: str, reasoning_lang: str,
                     reasoning_model: bool, force_english: bool = False) -> str:
    actual_lang = "en" if force_english else reasoning_lang
    messages = []
    if not reasoning_model:
        messages.append({"role": "system", "content": prompts.SYSTEM_PROMPT_COT})
    messages.append({"role": "user", "content": prompt})

    if reasoning_model:
        text = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)
        trigger = prompts.COT_THINK_TRIGGERS.get(actual_lang, "")
        if trigger:
            text += f"<think>\n{trigger}"
        return text

    if actual_lang != "en":
        cot_hint = f"Let's think step by step. Respond in {prompts.get_lang_name(actual_lang)}."
        forcing = prompts.COT_TRIGGERS.get(actual_lang, "")
    else:
        cot_hint, forcing = "Let's think step by step.", ""
    messages[-1]["content"] = f"{prompt}\n\n{cot_hint}"
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + forcing


def parse_mode(mode_str: str):
    parts = mode_str.strip().replace("_", "-").split("-")
    if len(parts) < 3:
        raise ValueError(f"Invalid mode: {mode_str}")
    q_lang, r_lang = parts[0], parts[1]
    skeleton = "-".join(parts[2:]).lower() in {"skeleton", "ske"}
    return q_lang, r_lang, skeleton


def normalize_modes(modes_raw: str) -> list:
    if modes_raw.strip().lower() == "all":
        all_langs = ["en", "es", "zh", "ko", "th", "sw", "te"]
        modes = []
        for lang in all_langs:
            modes.append(f"{lang}-{lang}-nonskeleton")
            if lang != "en":
                modes.append(f"{lang}-en-nonskeleton")
                modes.append(f"en-{lang}-nonskeleton")
        return modes
    normed = set()
    for m in modes_raw.split(","):
        m = m.strip()
        if not m:
            continue
        q, r, ske = parse_mode(m)
        normed.add(f"{q}-{r}-{'skeleton' if ske else 'nonskeleton'}")
    return list(normed)


# ═══════════════════════════════════════════════════════════════════════════════
#                           SAMPLING-PARAMETER FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def make_solver_sampling(args) -> SamplingParams:
    """Solver/answer sampling params, driven by --decoding and --rollout."""
    if args.decoding == "greedy":
        if args.rollout > 1:
            print(f"⚠️ --decoding greedy ignores --rollout {args.rollout}; using a single solution.")
        return SamplingParams(temperature=0.0, top_p=1.0, top_k=args.top_k,
                              max_tokens=args.max_tokens, n=1)
    return SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                          max_tokens=args.max_tokens, n=max(1, args.rollout))


def make_skeleton_sampling(args) -> SamplingParams:
    """Skeleton generation is always greedy/deterministic (matches released data)."""
    return SamplingParams(temperature=0.0, top_p=1.0, top_k=args.top_k,
                          max_tokens=args.max_tokens, n=1)


# ═══════════════════════════════════════════════════════════════════════════════
#                              EVALUATION RUNNERS
# ═══════════════════════════════════════════════════════════════════════════════

def _generate(llm, prompt_list, sp, lora):
    if lora:
        return llm.generate(prompt_list, sp, use_tqdm=False,
                            lora_request=LoRARequest("adapter", 1, lora))
    return llm.generate(prompt_list, sp, use_tqdm=False)


def run_skeleton(args, llm, tok, config, spec, full_rows, gen_sp, sk_sp, load_targets, langs):
    """Skeleton-based multi-turn reasoning (Turn 1 skeleton -> Turn 2 solve)."""
    global_idx = 0
    with open(args.output, "w", encoding="utf-8") as f_out:
        for target in load_targets:
            curr_lang, curr_split = target["lang"], target["split"]
            difficulty = curr_split if config["type"] == "polymath" else None
            print(f"\n📥 Loading [{args.dataset}] lang={curr_lang}, split={curr_split} ...")
            try:
                rows = select_local_rows(full_rows, spec, curr_lang, difficulty)
                rows = sample_rows(rows, args.sample_ratio, args.seed)
                print(f"✅ Loaded {len(rows)} samples.")

                task_queue = []
                for i, row in enumerate(rows):
                    q_lang = curr_lang or "en"
                    if q_lang not in langs:
                        continue
                    question_text = get_qa_text(row, q_lang)
                    if not question_text or not question_text.strip():
                        continue
                    task_queue.append({
                        "unique_id": row.get("id", i),
                        "question": question_text,
                        "answer": row.get("answer", ""),
                        "difficulty": row.get("difficulty", curr_split),
                        "lang": q_lang,
                    })
                if not task_queue:
                    print("⚠️ No valid tasks found.")
                    continue
                print(f"🚀 {len(task_queue)} tasks | skeleton_lang={args.skeleton_lang} "
                      f"| decoding={args.decoding} | rollout={gen_sp.n}")

                for batch in tqdm(chunks(task_queue, args.batch),
                                  total=(len(task_queue) // args.batch + 1)):
                    questions = [t["question"] for t in batch]
                    translated_q = [None] * len(batch)

                    # [Turn 0] optional query translation (ell_q = en)
                    if args.translate_q:
                        idxs = [i for i, t in enumerate(batch) if t["lang"] != "en"]
                        if idxs:
                            tp = [build_translation_prompt(tok, batch[i]["question"], batch[i]["lang"], "en")
                                  for i in idxs]
                            for i, out in zip(idxs, _generate(llm, tp, sk_sp, args.lora)):
                                if out.outputs:
                                    translated_q[i] = out.outputs[0].text.strip()
                                    questions[i] = translated_q[i]

                    # [Turn 1] skeleton generation (greedy)
                    sk_query_lang = "en" if args.translate_q else None
                    sk_prompts = [build_skeleton_prompt(tok, sk_query_lang or t["lang"], q, args.skeleton_lang)
                                  for t, q in zip(batch, questions)]
                    sk_outputs = _generate(llm, sk_prompts, sk_sp, args.lora)
                    skeletons = [out.outputs[0].text.strip() if out.outputs else "" for out in sk_outputs]

                    # [Turn 2] solve (greedy or sampling x rollout)
                    sol_prompts, mapping = [], []
                    for bi, (t, q, sk) in enumerate(zip(batch, questions, skeletons)):
                        if not sk:
                            continue
                        force_en = args.translate_cot and t["lang"] != "en"
                        sol_prompts.append(build_solver_prompt(tok, q, sk, t["lang"], force_english=force_en))
                        mapping.append(bi)

                    responses = [[] for _ in batch]
                    if sol_prompts:
                        for bi, out in zip(mapping, _generate(llm, sol_prompts, gen_sp, args.lora)):
                            if out.outputs:
                                responses[bi] = [o.text.strip() for o in out.outputs]

                    for i, t in enumerate(batch):
                        f_out.write(json.dumps({
                            "global_id": global_idx,
                            "original_id": t["unique_id"],
                            "prompt": t["question"],
                            "translated_question": translated_q[i],
                            "skeleton": [skeletons[i]] if skeletons[i] else [""],
                            "skeleton_lang": args.skeleton_lang,
                            "responses": responses[i] if responses[i] else [""],
                            "question_language": t["lang"],
                            "difficulty": t["difficulty"],
                            "answer": t["answer"],
                            "method": "skeleton_multiturn",
                            "translate_q": args.translate_q,
                            "translate_cot": args.translate_cot,
                            "decoding": args.decoding,
                        }, ensure_ascii=False) + "\n")
                        global_idx += 1
                    f_out.flush()
            except Exception as e:
                import traceback
                print(f"❌ Error: {e}")
                traceback.print_exc()
    print(f"\n🎉 Done! Saved to {args.output}")


def run_skeleton_from_existing(args, llm, tok, gen_sp):
    """Re-solve from pre-generated skeletons (e.g. multi-rollout over a greedy run)."""
    print(f"📖 Reading skeletons: {args.existing_skeletons}")
    records = []
    with open(args.existing_skeletons, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"✅ Loaded {len(records)} records | decoding={args.decoding} | rollout={gen_sp.n}")

    with open(args.output, "w", encoding="utf-8") as f_out:
        for batch in tqdm(chunks(records, args.batch), total=(len(records) // args.batch + 1)):
            sol_prompts, mapping = [], []
            for bi, item in enumerate(batch):
                sk_list = item.get("skeleton", [])
                sk = sk_list[0] if sk_list else ""
                if not sk:
                    continue
                q_lang = item.get("question_language", "en")
                force_en = args.translate_cot and q_lang != "en"
                sol_prompts.append(build_solver_prompt(tok, item["prompt"], sk, q_lang, force_english=force_en))
                mapping.append(bi)

            responses = [[] for _ in batch]
            if sol_prompts:
                for bi, out in zip(mapping, _generate(llm, sol_prompts, gen_sp, args.lora)):
                    if out.outputs:
                        responses[bi] = [o.text.strip() for o in out.outputs]

            for i, item in enumerate(batch):
                item["responses"] = responses[i] if responses[i] else [""]
                item["decoding"] = args.decoding
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            f_out.flush()
    print(f"\n🎉 Done! Saved to {args.output}")


def run_cot(args, llm, tok, config, spec, full_rows, gen_sp, sk_sp, load_targets, langs):
    """Direct CoT reasoning with optional query/answer translation."""
    mode_list = normalize_modes(args.modes)
    global_idx = 0
    with open(args.output, "w", encoding="utf-8") as f_out:
        for target in load_targets:
            curr_lang, curr_split = target["lang"], target["split"]
            difficulty = curr_split if config["type"] == "polymath" else None
            print(f"\n📥 Loading [{args.dataset}] lang={curr_lang}, split={curr_split} ...")
            try:
                rows = select_local_rows(full_rows, spec, curr_lang, difficulty)
                rows = sample_rows(rows, args.sample_ratio, args.seed)
                print(f"✅ Loaded {len(rows)} samples.")

                all_tasks = []
                for i, row in enumerate(rows):
                    for mode in mode_list:
                        q_lang, r_lang, _ = parse_mode(mode)
                        if curr_lang and q_lang != curr_lang:
                            continue
                        question_text = get_qa_text(row, q_lang)
                        if not question_text or not question_text.strip():
                            continue
                        all_tasks.append({
                            "unique_id": row.get("id", i),
                            "prompt": question_text,
                            "question_language": q_lang,
                            "reasoning_language": r_lang,
                            "mode": mode,
                            "difficulty": row.get("difficulty", curr_split),
                            "answer": row.get("answer", None),
                        })
                if not all_tasks:
                    print("⚠️ No tasks match. Skipping...")
                    continue
                print(f"🚀 {len(all_tasks)} tasks | decoding={args.decoding} | rollout={gen_sp.n}")

                for batch in tqdm(chunks(all_tasks, args.batch),
                                  total=(len(all_tasks) // args.batch + 1)):
                    questions = [t["prompt"] for t in batch]
                    translated_q = [None] * len(batch)

                    if args.translate_q:
                        idxs = [i for i, t in enumerate(batch) if t["question_language"] != "en"]
                        if idxs:
                            tp = [build_translation_prompt(tok, batch[i]["prompt"],
                                                           batch[i]["question_language"], "en") for i in idxs]
                            for i, out in zip(idxs, _generate(llm, tp, sk_sp, args.lora)):
                                if out.outputs:
                                    translated_q[i] = out.outputs[0].text.strip()
                                    questions[i] = translated_q[i]

                    cot_prompts = []
                    for t, q in zip(batch, questions):
                        force_en = args.translate_cot and t["reasoning_language"] != "en"
                        eff_lang = "en" if force_en else t["reasoning_language"]
                        cot_prompts.append(build_cot_prompt(tok, q, eff_lang, args.reasoning_model, force_en))

                    responses = [[o.text.strip() for o in out.outputs] if out.outputs else []
                                 for out in _generate(llm, cot_prompts, gen_sp, args.lora)]

                    for i, t in enumerate(batch):
                        f_out.write(json.dumps({
                            "global_id": global_idx,
                            "original_id": t["unique_id"],
                            "prompt": t["prompt"],
                            "translated_question": translated_q[i],
                            "responses": responses[i],
                            "question_language": t["question_language"],
                            "reasoning_language": t["reasoning_language"],
                            "mode": t["mode"],
                            "difficulty": t["difficulty"],
                            "answer": t["answer"],
                            "method": "cot",
                            "translate_q": args.translate_q,
                            "translate_cot": args.translate_cot,
                            "decoding": args.decoding,
                        }, ensure_ascii=False) + "\n")
                        global_idx += 1
                    f_out.flush()
            except Exception as e:
                import traceback
                print(f"❌ Error: {e}")
                traceback.print_exc()
    print(f"\n🎉 Done! Saved to {args.output}")


# ═══════════════════════════════════════════════════════════════════════════════
#                                   MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    if not args.dataset and not args.existing_skeletons:
        raise SystemExit("Provide --dataset (fresh run) or --existing_skeletons (reuse run).")

    print("=" * 70)
    print(f"🎯 method={args.method} | translate_q={args.translate_q} | "
          f"skeleton_lang={args.skeleton_lang} | decoding={args.decoding} | rollout={args.rollout}")
    print("=" * 70)

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    diffs = [d.strip() for d in args.difficulties.split(",") if d.strip()]

    # 7B Qwen fits on 4 GPUs; otherwise honor --tp
    tp = 4 if "qwen" in args.model.lower() and "7b" in args.model.lower() else args.tp

    print("🔧 Loading LLM...")
    llm = LLM(model=args.model, trust_remote_code=True, tensor_parallel_size=tp,
              gpu_memory_utilization=args.gpu_mem, dtype=args.dtype, hf_token=HF_TOKEN)
    tok = llm.get_tokenizer()

    gen_sp = make_solver_sampling(args)
    sk_sp = make_skeleton_sampling(args)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Reuse pre-generated skeletons -> only re-solve
    if args.existing_skeletons:
        run_skeleton_from_existing(args, llm, tok, gen_sp)
        return

    config = get_dataset_config(args.dataset)
    spec_key = resolve_local_spec_key(args.dataset)
    spec = LOCAL_SPECS[spec_key]
    data_root = args.data_root or os.path.join(
        os.environ.get("LASEF_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "test_data")
    full_rows = load_local_file(data_root, spec)
    print(f"🛠️ Dataset type: {config['type']} | local file: {spec['file']} ({len(full_rows)} rows)")
    load_targets = build_load_targets(config, langs, diffs)

    if args.method == "skeleton":
        run_skeleton(args, llm, tok, config, spec, full_rows, gen_sp, sk_sp, load_targets, langs)
    else:
        run_cot(args, llm, tok, config, spec, full_rows, gen_sp, sk_sp, load_targets, langs)


if __name__ == "__main__":
    main()
