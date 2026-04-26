# TaskReasoner

TaskReasoner is a task understanding and prediction system for software project management.

It is built on the open-source TAWOS dataset and combines task text with structured metadata to support:

- task type classification
- coarse priority classification
- difficulty classification
- task relation classification
- story point regression

## Main Components

- `src/`: classic TF-IDF and scikit-learn training / prediction scripts
- `attention_training/`: cleaned-data preparation, TF-IDF task heads, MiniLM embedding heads, and transformer fine-tuning
- `scripts/`: setup, validation, import, and export helpers
- `思路/`: local validation notes and experiment writeups

## Current Best Models

### Task heads

Best lightweight task model bundle:

- path: `attention_training/models/tfidf_task_heads_strategy_v3_random50/`
- features: TF-IDF + one-hot structured metadata
- models:
  - `task_type`: balanced `LinearSVC`
  - `priority`: balanced `LogisticRegression`
  - `difficulty_label`: balanced `LogisticRegression`

Performance:

| Target | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| `task_type` | 0.8330 | 0.6977 | 0.8344 |
| `priority` | 0.7189 | 0.6818 | 0.7208 |
| `difficulty_label` | 0.6490 | 0.6283 | 0.6468 |

### Relation model

Best relation model bundle:

- path: `attention_training/models/embedding_heads_strategy_v2_random30_relation_polarity3/`
- features: MiniLM embeddings + structured pair features
- labels: `not_related`, `positive_related`, `negative_related`

Performance:

| Target | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| `task_relation` | 0.8751 | 0.7589 | 0.8712 |

## Install

Base dependencies:

```bash
python -m pip install -r requirements.txt
```

Transformer / MiniLM experiments:

```bash
python -m pip install -r attention_training/requirements-attention.txt
```

## Notes

This repository is source-focused. Large local-only artifacts such as raw data, processed exports, model checkpoints, and MySQL sandbox files should stay out of GitHub.

See:

- `data/README.md`
- `models/README.md`
- `attention_training/data/README.md`
- `attention_training/models/README.md`
