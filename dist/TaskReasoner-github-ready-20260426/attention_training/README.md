# Attention Training

This folder is independent from the existing TF-IDF models. It prepares cleaned TAWOS data for:

1. masked-language-model pretraining on unlabeled task text;
2. Sentence-BERT embedding extraction plus scikit-learn heads;
3. supervised transformer fine-tuning for one target at a time.

## Install Extra Dependencies

```bash
/opt/anaconda3/bin/python -m pip install -r attention_training/requirements-attention.txt
```

## Prepare Clean Data

The clean labels are generated from the local TAWOS export:

- `priority_coarse`: `high`, `medium`, `low`.
- `difficulty_label_derived`: generated from cleaned `story_point`.
- `story_point_clean`: positive story points with values above 100 dropped.
- `story_point_bin5`: five difficulty bands.
- weak priority labels: when original priority is missing, use cleaned story point first and text keywords second.

```bash
/opt/anaconda3/bin/python attention_training/prepare_attention_data.py \
  --max-task-rows 120000 \
  --max-link-rows 60000 \
  --story-point-max 100 \
  --link-negative-ratio 1.0 \
  --output-dir attention_training/data/clean30
```

## Strategy V2: Cleaner Labels And Two-Stage Relations

This version uses the stricter plan:

- `task_type_model`: keep the top 8 task types and map the rest to `Other`.
- `priority_coarse`: keep `high`, `medium`, `low`.
- `difficulty_label_derived`: require multi-source agreement from story point, effort, issue type, resolution time, and original difficulty when available.
- relation labels: add `relation_binary` for not-related vs `has_relation`, then train relation type only for positive pairs.
- optional coarse relation labels: use `--relation-label-mode coarse4` to collapse all positive relation labels into `depends`, `blocks`, and `related`, while keeping `no_relation`.
- optional polarity relation labels: use `--relation-label-mode polarity3` to collapse relation labels into `not_related`, `positive_related`, and `negative_related`.
- structured features: project, issue type, priority, status, resolution, components, and pair graph features are exported for model heads.

```bash
/opt/anaconda3/bin/python attention_training/prepare_attention_data.py \
  --max-task-rows 0 \
  --max-link-rows 0 \
  --task-sample-frac 0.7 \
  --link-sample-frac 0.7 \
  --task-type-top-n 8 \
  --difficulty-label-mode consensus \
  --difficulty-min-agree 2 \
  --difficulty-balance-ratio 2.0 \
  --story-point-max 100 \
  --link-negative-ratio 1.0 \
  --output-dir attention_training/data/strategy_v2_random70
```

Four-class relation variant:

```bash
/opt/anaconda3/bin/python attention_training/prepare_attention_data.py \
  --max-task-rows 0 \
  --max-link-rows 0 \
  --task-sample-frac 0.3 \
  --link-sample-frac 0.3 \
  --task-type-top-n 8 \
  --difficulty-label-mode consensus \
  --difficulty-min-agree 2 \
  --difficulty-balance-ratio 2.0 \
  --story-point-max 100 \
  --link-negative-ratio 1.0 \
  --relation-label-mode coarse4 \
  --output-dir attention_training/data/strategy_v2_random30_relation_coarse4
```

Three-class polarity relation variant:

```bash
/opt/anaconda3/bin/python attention_training/prepare_attention_data.py \
  --max-task-rows 0 \
  --max-link-rows 0 \
  --task-sample-frac 0.3 \
  --link-sample-frac 0.3 \
  --task-type-top-n 8 \
  --difficulty-label-mode consensus \
  --difficulty-min-agree 2 \
  --difficulty-balance-ratio 2.0 \
  --story-point-max 100 \
  --link-negative-ratio 1.0 \
  --relation-label-mode polarity3 \
  --output-dir attention_training/data/strategy_v2_random30_relation_polarity3
```

Embedding-head baseline with structured features and two-stage relation:

```bash
caffeinate -dimsu /opt/anaconda3/bin/python attention_training/train_embedding_heads.py \
  --data-dir attention_training/data/strategy_v2_random70 \
  --model-name sentence-transformers/all-MiniLM-L6-v2 \
  --batch-size 128 \
  --max-text-chars 1800 \
  --max-seq-length 192 \
  --use-structured-features \
  --two-stage-link \
  --oversample-minority \
  --output-dir attention_training/models/embedding_heads_strategy_v2_random70
```

Supervised MiniLM fine-tuning examples:

```bash
/opt/anaconda3/bin/python attention_training/train_transformer_finetune.py \
  --data-dir attention_training/data/strategy_v2_random70 \
  --task task_type \
  --base-model sentence-transformers/all-MiniLM-L6-v2 \
  --class-weight balanced \
  --output-dir attention_training/models/finetuned_strategy_v2

/opt/anaconda3/bin/python attention_training/train_transformer_finetune.py \
  --data-dir attention_training/data/strategy_v2_random70 \
  --task task_relation_binary \
  --base-model sentence-transformers/all-MiniLM-L6-v2 \
  --class-weight balanced \
  --output-dir attention_training/models/finetuned_strategy_v2

/opt/anaconda3/bin/python attention_training/train_transformer_finetune.py \
  --data-dir attention_training/data/strategy_v2_random70 \
  --task task_relation_type \
  --base-model sentence-transformers/all-MiniLM-L6-v2 \
  --class-weight balanced \
  --output-dir attention_training/models/finetuned_strategy_v2
```

## Optional MLM Continued Pretraining

```bash
/opt/anaconda3/bin/python attention_training/train_mlm.py \
  --corpus attention_training/data/clean30/unlabeled_corpus.txt \
  --base-model distilbert-base-uncased \
  --output-dir attention_training/models/tawos_mlm_clean30 \
  --max-lines 50000 \
  --epochs 1 \
  --batch-size 8
```

The output directory can then be used as `--base-model` for supervised fine-tuning.

## Embedding Baseline

```bash
/opt/anaconda3/bin/python attention_training/train_embedding_heads.py \
  --data-dir attention_training/data/clean30 \
  --model-name sentence-transformers/all-MiniLM-L6-v2 \
  --output-dir attention_training/models/embedding_heads_clean30
```

To retrain only the relation head after adding `no_relation` samples:

```bash
/opt/anaconda3/bin/python attention_training/train_embedding_heads.py \
  --data-dir attention_training/data/clean30_neg \
  --model-name sentence-transformers/all-MiniLM-L6-v2 \
  --batch-size 128 \
  --max-text-chars 1800 \
  --max-seq-length 192 \
  --only-link \
  --output-dir attention_training/models/embedding_heads_clean30_neg_link
```

## Supervised Transformer Fine-Tuning

Examples:

```bash
/opt/anaconda3/bin/python attention_training/train_transformer_finetune.py \
  --data-dir attention_training/data/clean30 \
  --task priority \
  --base-model distilbert-base-uncased \
  --output-dir attention_training/models/finetuned_clean30

/opt/anaconda3/bin/python attention_training/train_transformer_finetune.py \
  --data-dir attention_training/data/clean30 \
  --task task_relation \
  --base-model distilbert-base-uncased \
  --output-dir attention_training/models/finetuned_clean30
```

Supported `--task` values:

```text
task_type
priority
difficulty_label
story_point_bin5
story_point_regression
task_relation
```
