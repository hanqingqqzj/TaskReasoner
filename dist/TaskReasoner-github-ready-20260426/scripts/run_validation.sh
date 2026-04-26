#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python}"
MODE="${1:-smoke}"

cd "$ROOT_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found: $PYTHON_BIN" >&2
  exit 1
fi

case "$MODE" in
  smoke)
    TASK_MODEL_DIR="models/task_model_smoke"
    LINK_MODEL_DIR="models/link_model_smoke"
    TASK_ARGS=(
      --max-rows 3000
      --max-features 3000
      --max-iter 5
      --min-class-count 5
      --max-classes-per-target 8
      --model-dir "$TASK_MODEL_DIR"
    )
    LINK_ARGS=(
      --max-tasks 5000
      --max-positive-links 3000
      --max-features 3000
      --max-iter 5
      --min-class-count 5
      --model-dir "$LINK_MODEL_DIR"
    )
    ;;
  standard)
    TASK_MODEL_DIR="models/task_model"
    LINK_MODEL_DIR="models/link_model"
    TASK_ARGS=(
      --max-rows 150000
      --max-features 50000
      --max-iter 50
      --model-dir "$TASK_MODEL_DIR"
    )
    LINK_ARGS=(
      --max-tasks 200000
      --max-positive-links 120000
      --max-features 60000
      --max-iter 50
      --model-dir "$LINK_MODEL_DIR"
    )
    ;;
  large)
    TASK_MODEL_DIR="models/task_model_large"
    LINK_MODEL_DIR="models/link_model_large"
    TASK_ARGS=(
      --max-rows 400000
      --max-features 70000
      --max-iter 80
      --model-dir "$TASK_MODEL_DIR"
    )
    LINK_ARGS=(
      --max-tasks 400000
      --max-positive-links 200000
      --max-features 70000
      --max-iter 70
      --model-dir "$LINK_MODEL_DIR"
    )
    ;;
  optimized)
    TASK_MODEL_DIR="models/task_model_optimized"
    LINK_MODEL_DIR="models/link_model_optimized"
    TASK_ARGS=(
      --max-rows 400000
      --max-features 70000
      --max-iter 80
      --algorithm auto
      --model-dir "$TASK_MODEL_DIR"
    )
    LINK_ARGS=(
      --max-tasks 400000
      --max-positive-links 200000
      --max-features 70000
      --max-iter 70
      --algorithm sgd_log_loss
      --model-dir "$LINK_MODEL_DIR"
    )
    ;;
  final)
    TASK_MODEL_DIR="models/task_model_final"
    LINK_MODEL_DIR="models/link_model_final"
    TASK_ARGS=(
      --max-rows 400000
      --max-features 70000
      --max-iter 80
      --hidden-size-1 128
      --hidden-size-2 64
      --algorithm final
      --train-story-point-regressor
      --model-dir "$TASK_MODEL_DIR"
    )
    LINK_ARGS=(
      --max-tasks 400000
      --max-positive-links 200000
      --max-features 70000
      --max-iter 70
      --hidden-size-1 256
      --hidden-size-2 128
      --algorithm mlp
      --model-dir "$LINK_MODEL_DIR"
    )
    ;;
  final30)
    TASK_MODEL_DIR="models/task_model_final_30pct"
    LINK_MODEL_DIR="models/link_model_final_30pct"
    TASK_ARGS=(
      --max-rows 120000
      --max-features 70000
      --max-iter 80
      --hidden-size-1 128
      --hidden-size-2 64
      --algorithm final
      --train-story-point-regressor
      --model-dir "$TASK_MODEL_DIR"
    )
    LINK_ARGS=(
      --max-tasks 120000
      --max-positive-links 60000
      --max-features 70000
      --max-iter 70
      --hidden-size-1 256
      --hidden-size-2 128
      --algorithm mlp
      --model-dir "$LINK_MODEL_DIR"
    )
    ;;
  clean30)
    TASK_MODEL_DIR="models/task_model_clean_30pct"
    LINK_MODEL_DIR="models/link_model_clean_30pct"
    TASK_ARGS=(
      --max-rows 120000
      --max-features 70000
      --max-iter 80
      --hidden-size-1 128
      --hidden-size-2 64
      --algorithm final
      --coarse-priority
      --derive-difficulty-from-story-point
      --story-point-max 100
      --story-point-bins five
      --train-story-point-regressor
      --model-dir "$TASK_MODEL_DIR"
    )
    LINK_ARGS=(
      --max-tasks 120000
      --max-positive-links 60000
      --max-features 70000
      --max-iter 70
      --hidden-size-1 256
      --hidden-size-2 128
      --algorithm mlp
      --model-dir "$LINK_MODEL_DIR"
    )
    ;;
  full)
    TASK_MODEL_DIR="models/task_model_full"
    LINK_MODEL_DIR="models/link_model_full"
    TASK_ARGS=(
      --max-rows 0
      --max-features 80000
      --max-iter 100
      --model-dir "$TASK_MODEL_DIR"
    )
    LINK_ARGS=(
      --max-tasks 0
      --max-positive-links 0
      --max-features 80000
      --max-iter 80
      --model-dir "$LINK_MODEL_DIR"
    )
    ;;
  *)
    echo "Usage: $0 [smoke|standard|large|optimized|final|final30|clean30|full]" >&2
    exit 1
    ;;
esac

echo "Running ${MODE} validation with ${PYTHON_BIN}"
echo
echo "Step 1/2: task attribute and difficulty model"
"$PYTHON_BIN" src/train_task_model.py "${TASK_ARGS[@]}"

echo
echo "Step 2/2: task relation model"
"$PYTHON_BIN" src/train_link_model.py "${LINK_ARGS[@]}"

echo
echo "Validation finished."
echo "Task metrics:"
cat "$TASK_MODEL_DIR/metrics.json"
if [[ -f "$TASK_MODEL_DIR/regressor_metrics.json" ]]; then
  echo
  echo "Task regressor metrics:"
  cat "$TASK_MODEL_DIR/regressor_metrics.json"
fi
echo
echo "Link metrics:"
cat "$LINK_MODEL_DIR/metrics.json"
