#!/bin/bash
# ==============================================================================
# LASEF — one-shot pipeline: evaluate across skeleton languages, then plot the
#         skeleton-language accuracy-delta heatmap.
#
#   bash LASEF.sh
#
# Everything is configured by the variables below (override via the environment):
#   MODEL          HF model id                       (default Qwen/Qwen2.5-7B-Instruct)
#   DATASET        local benchmark key                (default mgsm)
#                  one of: mgsm | mgsm_low | math500 | polymath
#   LANGS          target/query languages ℓa         (default en,zh,es,ko,th,sw,te)
#   SKELETON_LANGS skeleton languages ℓs to sweep     (default en zh es ru ko th)
#   ROLLOUT        N>1 -> sampling rollouts; 1 -> greedy single  (default 1)
#   TRANSLATE_Q    "1" -> translate query to English (ℓq=en)     (default "")
#   OUT_DIR        where result .jsonl files go
#   FIG            heatmap output PDF
#   CUDA_VISIBLE_DEVICES   GPUs to use                (default 0,1,2,3)
#
# Note: "en" is the baseline skeleton language and is always included so the
# heatmap (other ℓs − English ℓs) can be computed.
# ==============================================================================
set -euo pipefail

# --- Project root (override with LASEF_ROOT, else auto-detect) ----------------
PRJ_PATH="${LASEF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export LASEF_ROOT="$PRJ_PATH"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EVAL="$PRJ_PATH/scripts/src/evaluate.py"
PLOT="$PRJ_PATH/analysis/plot_heatmap.py"

# --- Configuration ------------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_NAME="$(basename "$MODEL")"
DATASET="${DATASET:-anonymous/MGSM}"
LANGS="${LANGS:-en,zh,es,ko,th,sw,te}"
SKELETON_LANGS="${SKELETON_LANGS:-en zh es ru ko th}"
ROLLOUT="${ROLLOUT:-1}"
TRANSLATE_Q="${TRANSLATE_Q:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# Make sure the baseline skeleton language (en) is present in the sweep.
case " $SKELETON_LANGS " in *" en "*) ;; *) SKELETON_LANGS="en $SKELETON_LANGS" ;; esac

# Decoding flags derived from ROLLOUT; the rollout count is encoded in the output
# folder name (e.g. single_rollout / 5-rollout) so plot_heatmap can pick it up,
# while each file keeps the direct skelLang-<ℓs>.jsonl name.
if [ "$ROLLOUT" -gt 1 ]; then
    DECODE_FLAGS=(--decoding sampling --rollout "$ROLLOUT" --temperature 0.7 --top_p 0.8)
    TAG="${ROLLOUT}-rollout"
else
    DECODE_FLAGS=(--decoding greedy)
    TAG="single_rollout"
fi
TQ_FLAGS=(); [ -n "$TRANSLATE_Q" ] && { TQ_FLAGS=(--translate_q); TAG="${TAG}_transQ"; }

OUT_DIR="${OUT_DIR:-$PRJ_PATH/data/results/run/$TAG}"
FIG="${FIG:-$PRJ_PATH/figure/${MODEL_NAME}_${TAG}_skeleton_delta_heatmap.pdf}"
mkdir -p "$OUT_DIR" "$(dirname "$FIG")"

echo "========================================================================"
echo " LASEF pipeline"
echo "   model=$MODEL   dataset=$DATASET"
echo "   skeleton_langs=[$SKELETON_LANGS]   rollout=$ROLLOUT   translate_q=${TRANSLATE_Q:-no}"
echo "   out_dir=$OUT_DIR"
echo "   GPUs=$CUDA_VISIBLE_DEVICES"
echo "========================================================================"

# --- Step 1: evaluate across skeleton languages -------------------------------
for SL in $SKELETON_LANGS; do
    OUT="$OUT_DIR/${MODEL_NAME}-skeleton_multiturn_skelLang-${SL}.jsonl"
    echo ""
    echo "▶ skeleton_lang=$SL  ->  $(basename "$OUT")"
    "$PYTHON_BIN" "$EVAL" \
        --method skeleton --model "$MODEL" --dataset "$DATASET" \
        --langs "$LANGS" --skeleton_lang "$SL" \
        "${DECODE_FLAGS[@]}" "${TQ_FLAGS[@]}" \
        --output "$OUT"
done

# --- Step 2: plot the skeleton-language delta heatmap -------------------------
echo ""
echo "▶ Plotting heatmap  ->  $FIG"
"$PYTHON_BIN" "$PLOT" \
    --data_dir "$OUT_DIR" \
    --model "$MODEL_NAME" \
    --output "$FIG"

echo ""
echo "🎉 Done. Heatmap saved to: $FIG"
