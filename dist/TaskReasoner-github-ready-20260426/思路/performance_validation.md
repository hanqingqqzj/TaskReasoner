# 模型运行与性能验证

## 1. 环境检查

本项目当前使用 Anaconda Python：

```bash
/opt/anaconda3/bin/python --version
/opt/anaconda3/bin/python -m pip install -r requirements.txt
```

当前模型主要使用：

```text
TF-IDF + MLPClassifier / LogisticRegression / Ridge
```

所以暂时不需要 PyTorch。

## 2. 快速验证

快速验证用于检查代码、数据、训练、保存、预测是否全部跑通。

```bash
cd /Users/zhaohanqing/Desktop/TaskReasoner
./scripts/run_validation.sh smoke
```

输出文件：

```text
models/task_model_smoke/task_model.joblib
models/task_model_smoke/metrics.json
models/link_model_smoke/link_model.joblib
models/link_model_smoke/metrics.json
```

## 3. 标准基线验证

标准验证使用更大样本和更多训练轮数，适合记录正式基线结果。

```bash
cd /Users/zhaohanqing/Desktop/TaskReasoner
./scripts/run_validation.sh standard
```

当前 `standard` 配置：

```text
task model:
  max rows: 150000
  max TF-IDF features: 50000
  max iterations: 50

link model:
  max task rows: 200000
  max positive links: 120000
  max TF-IDF features: 60000
  max iterations: 50
```

输出文件：

```text
models/task_model/task_model.joblib
models/task_model/metrics.json
models/link_model/link_model.joblib
models/link_model/metrics.json
```

## 4. 大样本验证

如果你想直接使用大部分数据训练，但又不想一上来就压满 full，可以先用 `large`。

```bash
cd /Users/zhaohanqing/Desktop/TaskReasoner
./scripts/run_validation.sh large
```

当前 `large` 配置：

```text
task model:
  max rows: 400000
  max TF-IDF features: 70000
  max iterations: 80

link model:
  max task rows: 400000
  max positive links: 200000
  max TF-IDF features: 70000
  max iterations: 70
```

输出文件：

```text
models/task_model_large/task_model.joblib
models/task_model_large/metrics.json
models/link_model_large/link_model.joblib
models/link_model_large/metrics.json
```

## 5. 全量验证

全量验证会使用全部可用数据，耗时和内存占用更高。

```bash
cd /Users/zhaohanqing/Desktop/TaskReasoner
./scripts/run_validation.sh full
```

当前 `full` 配置：

```text
task model:
  max rows: all
  max TF-IDF features: 80000
  max iterations: 100

link model:
  max task rows: all
  max positive links: all
  max TF-IDF features: 80000
  max iterations: 80
```

如果电脑卡顿，先停止 full，优先使用 standard。全量训练会覆盖并保存到：

```text
models/task_model_full/
models/link_model_full/
```

单独运行训练脚本时，现在默认也是全量训练：

```bash
/opt/anaconda3/bin/python src/train_task_model.py
/opt/anaconda3/bin/python src/train_link_model.py
```

如果只是调试，请显式传入较小参数，例如：

```bash
/opt/anaconda3/bin/python src/train_task_model.py --max-rows 3000 --max-iter 5
```

## 6. 优化算法验证

如果要使用当前验证出的更快算法，运行：

```bash
cd /Users/zhaohanqing/Desktop/TaskReasoner
./scripts/run_validation.sh optimized
```

当前 `optimized` 配置：

```text
task model:
  data volume: same as large
  algorithm: auto
    task_type -> SGD log-loss
    priority -> balanced SGD log-loss
    difficulty_label -> ComplementNB

link model:
  data volume: same as large
  algorithm: SGD log-loss
```

输出文件：

```text
models/task_model_optimized/task_model.joblib
models/task_model_optimized/metrics.json
models/link_model_optimized/link_model.joblib
models/link_model_optimized/metrics.json
```

## 7. 验证过程做了什么

如果要运行最终选定方案，使用：

```bash
cd /Users/zhaohanqing/Desktop/TaskReasoner
./scripts/run_validation.sh final
```

当前 `final` 配置：

```text
task model:
  data volume: same as large
  task_type -> MLP(128, 64)
  priority -> balanced Logistic Regression
  difficulty_label -> MLP(128, 64)
  story_point -> Ridge regression on log1p(story_point)

link model:
  data volume: same as large
  task_relation -> direct MLP(256, 128)
```

输出文件：

```text
models/task_model_final/task_model.joblib
models/task_model_final/metrics.json
models/task_model_final/regressor_metrics.json
models/link_model_final/link_model.joblib
models/link_model_final/metrics.json
```

任务模型：

```text
title + description_text
  -> TF-IDF
  -> 两层 MLP
  -> task_type / priority / difficulty_label
```

依赖模型：

```text
source task + target task
  -> TF-IDF
  -> 两层 MLP
  -> relation label
```

两个训练脚本都会自动做：

```text
80% training set
20% test set
```

测试集不会参与训练，只用于计算最终指标。

## 8. 指标怎么看

`accuracy`：

```text
正确预测数量 / 测试集总数量
```

适合快速看整体正确率，但如果类别不平衡，会偏向大类。

`macro_f1`：

```text
每个类别的 F1 分数取平均
```

它更重视小类。如果 `accuracy` 高但 `macro_f1` 低，说明模型可能只学会了预测大类。

`weighted_f1`：

```text
按每个类别样本数加权后的 F1
```

它比 accuracy 更细，但仍然会受大类影响。

## 9. 当前 smoke test 结果解读

当前快速验证结果大致是：

```text
task_type accuracy: 约 0.74
priority accuracy: 约 0.83
difficulty accuracy: 约 0.54
link relation accuracy: 约 0.52
```

这些不是最终性能，只说明端到端流程跑通。因为 smoke test 只用了小样本和 5 轮训练。

更值得关注的是：

```text
macro_f1
```

如果 macro_f1 很低，说明模型对少数类别识别不好。TAWOS 数据类别非常不平衡，这是正常初期现象。

## 10. 如何判断模型是否真的变好

每次改模型后，固定同一套验证命令，例如：

```bash
./scripts/run_validation.sh standard
```

然后比较：

```text
models/task_model/metrics.json
models/link_model/metrics.json
```

优先看：

```text
macro_f1 是否提升
weighted_f1 是否提升
accuracy 是否没有明显下降
```

如果只提升 accuracy，但 macro_f1 下降，通常不是好改动。

## 11. 运行单个预测

任务属性/难度预测：

```bash
/opt/anaconda3/bin/python src/predict_task.py \
  --model models/task_model/task_model.joblib \
  --title "Add validation for empty task descriptions" \
  --description "Reject empty input, show a clear error, and add tests for invalid task text."
```

任务关系预测：

```bash
/opt/anaconda3/bin/python src/predict_link.py \
  --model models/link_model/link_model.joblib \
  --source-title "Create database schema" \
  --source-description "Define tables and indexes for task records." \
  --target-title "Build training pipeline" \
  --target-description "Read task records from the database and train the model."
```

## 12. 下一步性能提升方向

优先级从高到低：

```text
1. 使用 standard 数据量建立稳定基线
2. 处理类别不平衡，提高 macro_f1
3. 加入 confusion matrix，观察错在哪里
4. 用 SentenceTransformer 替代 TF-IDF
5. 用 PyTorch 做真正共享隐藏层的多任务模型
```
