Neural Task Simulation and Prediction
This repository contains several training pipelines for task understanding on the TAWOS Jira dataset:

task type classification
coarse priority classification
difficulty classification
task relation classification
story point regression
The project started from scikit-learn TF-IDF baselines and later expanded to transformer-based experiments with MiniLM and supervised fine-tuning.

What This Project Does
The code supports three main workflows:

prepare and clean task data exported from TAWOS
train classic text models such as MLP, LinearSVC, Logistic Regression, SGD, and Ridge
train embedding-based and transformer-based models for stronger relation and text understanding experiments
Repository Layout
src/                  classic task and link training / prediction scripts
scripts/              data setup, import, export, and validation helpers
attention_training/   cleaned-data preparation, TF-IDF heads, MiniLM heads, transformer fine-tuning
思路/                 local notes and validation writeups
data/                 local dataset workspace (not tracked in GitHub package)
models/               local trained models (not tracked in GitHub package)
Main Pipelines
1. Classic TF-IDF / scikit-learn pipeline
Core scripts:

src/train_task_model.py
src/train_link_model.py
src/predict_task.py
src/predict_link.py
These scripts implement the earlier baseline models for:

task_type
priority
difficulty_label
task_relation
2. Clean-data + TF-IDF optimized task heads
Core scripts:

attention_training/prepare_attention_data.py
attention_training/train_tfidf_task_heads.py
This is the current best lightweight pipeline for:

task_type
priority
difficulty_label
It uses:

cleaned labels from TAWOS exports
TF-IDF text features
one-hot encoded structured metadata
class-balanced linear classifiers
3. MiniLM / transformer experiments
Core scripts:

attention_training/train_embedding_heads.py
attention_training/train_transformer_finetune.py
attention_training/train_mlm.py
These scripts support:

Sentence-BERT embedding extraction
embedding + classifier heads
supervised transformer fine-tuning
optional masked-language-model continued pretraining
The best task relation result so far comes from this branch.

Data
This GitHub-ready package does not include the full dataset, MySQL sandbox, processed TSV exports, or large model artifacts.

Local dataset sizes on the original machine were approximately:

tasks.tsv: 458,232 task rows
task_links.tsv: 246,577 link rows
Large local-only directories are intentionally excluded from the upload package:

data/raw/
data/processed/
data/mysql_sandbox/
models/
attention_training/data/
attention_training/models/
See:

data/README.md
models/README.md
attention_training/data/README.md
attention_training/models/README.md
Environment
Base dependencies:

python -m pip install -r requirements.txt
Transformer / MiniLM experiments:

python -m pip install -r attention_training/requirements-attention.txt
Base package list:

numpy
pandas
scikit-learn
joblib
Extra transformer packages:

torch
transformers
sentence-transformers
accelerate
Typical Workflow
Step 1: Prepare cleaned data
Example:

python attention_training/prepare_attention_data.py \
  --max-task-rows 0 \
  --max-link-rows 0 \
  --task-sample-frac 0.5 \
  --task-type-top-n 8 \
  --difficulty-label-mode consensus \
  --difficulty-min-agree 2 \
  --difficulty-balance-ratio 2.0 \
  --story-point-max 100 \
  --link-negative-ratio 1.0 \
  --relation-label-mode polarity3 \
  --output-dir attention_training/data/strategy_v3_random50_task_optimized
Step 2: Train optimized TF-IDF task heads
Example:

python attention_training/train_tfidf_task_heads.py \
  --data-dir attention_training/data/strategy_v3_random50_task_optimized \
  --output-dir attention_training/models/tfidf_task_heads_strategy_v3_random50 \
  --candidate-profile compact \
  --max-features 180000
Step 3: Train relation model
Example:

python attention_training/train_embedding_heads.py \
  --data-dir attention_training/data/strategy_v2_random30_relation_polarity3 \
  --model-name sentence-transformers/all-MiniLM-L6-v2 \
  --batch-size 128 \
  --max-text-chars 1800 \
  --max-seq-length 192 \
  --use-structured-features \
  --two-stage-link \
  --oversample-minority \
  --only-link \
  --output-dir attention_training/models/embedding_heads_strategy_v2_random30_relation_polarity3
Current Best Reported Results
Optimized TF-IDF task heads
Using 50% prepared task data:

Target	Model	Accuracy	Macro F1	Weighted F1
task_type	TF-IDF + One-Hot + balanced LinearSVC	0.8330	0.6977	0.8344
priority	TF-IDF + One-Hot + balanced Logistic Regression	0.7189	0.6818	0.7208
difficulty_label	TF-IDF + One-Hot + balanced Logistic Regression	0.6490	0.6283	0.6468
Relation model
Using 30% prepared relation data:

Target	Model	Accuracy	Macro F1	Weighted F1
task_relation	MiniLM embeddings + structured features + two-stage relation head	0.8751	0.7589	0.8712
Validation Helpers
Useful scripts:

scripts/run_validation.sh
scripts/compare_algorithms.py
scripts/validate_mlp_plan.py
scripts/validate_final_plan.py
Project notes:

思路/performance_validation.md
思路/step_by_step.md
Upload Notes
Before pushing to GitHub, the repository should stay source-focused:

keep code, scripts, requirements, and documentation
exclude local dataset dumps
exclude trained model binaries and checkpoints
exclude IDE folders, virtual environments, caches, and .DS_Store
This packaging cleanup is already reflected in the generated .gitignore.
