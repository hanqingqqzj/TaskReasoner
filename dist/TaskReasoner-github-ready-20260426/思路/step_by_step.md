# Step-by-Step 思路与公式

## 1. 问题拆分

这个项目先拆成两个模型。

模型 A：任务属性与难度判断。

```text
输入: title + description_text
输出: task_type, priority, difficulty_label
```

模型 B：任务关系/依赖判断。

```text
输入: source task text + target task text
输出: relation label
```

这样做的原因是：单任务判断只看一个任务文本，而依赖关系判断必须同时看两个任务。

## 2. 文本向量化：TF-IDF

原始文本不能直接进入 MLP，所以先把文本变成向量。

词频：

```text
tf(t, d) = count(t in d)
```

逆文档频率：

```text
idf(t) = log((1 + N) / (1 + df(t))) + 1
```

TF-IDF：

```text
x_t = tf(t, d) * idf(t)
```

其中：

- `N` 是文档总数
- `df(t)` 是包含词 `t` 的文档数
- `x_t` 是文本在词 `t` 维度上的数值

直觉：常出现在当前任务、但不在所有任务里都泛滥的词，权重更高。

## 3. 两层 MLP

输入向量：

```text
x ∈ R^d
```

第一层：

```text
h1 = ReLU(W1 x + b1)
```

第二层：

```text
h2 = ReLU(W2 h1 + b2)
```

输出层：

```text
z = W3 h2 + b3
```

其中 ReLU 是：

```text
ReLU(a) = max(0, a)
```

直觉：第一层学习粗粒度文本模式，第二层组合这些模式，输出层把它们映射成类别。

## 4. 分类概率：Softmax

对每个类别 `k`：

```text
p(y = k | x) = exp(z_k) / Σ_j exp(z_j)
```

输出不是单纯的类别名，而是每个类别的概率。例如：

```text
Bug: 0.72
Task: 0.18
Story: 0.06
```

## 5. 损失函数：交叉熵

单个分类头的损失：

```text
L = -Σ_k y_k log(p_k)
```

如果真实类别是 `Bug`，那么 `y_bug = 1`，其他类别为 0。模型越确信正确类别，损失越小。

## 6. 多个任务属性怎么处理

当前第一版实现中，`task_type`、`priority`、`difficulty_label` 各训练一个两层 MLP。

```text
text -> TF-IDF -> MLP_type       -> task_type
text -> TF-IDF -> MLP_priority   -> priority
text -> TF-IDF -> MLP_difficulty -> difficulty_label
```

下一版可以升级为真正共享隐藏层：

```text
text -> shared encoder -> shared h2 -> type head
                                  -> priority head
                                  -> difficulty head
```

共享模型的总损失是：

```text
L_total = L_type + L_priority + L_difficulty
```

如果加入时间预测：

```text
L_total = L_type + L_priority + L_difficulty + λ L_time
```

时间预测常用均方误差：

```text
L_time = (y_time - ŷ_time)^2
```

## 7. 难度标签从哪里来

TAWOS 的 `Story_Point` 可以近似代表任务难度。当前导出时使用：

```text
Story_Point <= 2      -> easy
3 <= Story_Point <= 5 -> medium
Story_Point > 5       -> hard
```

注意：Story Point 比实际耗时更接近“难度”。实际耗时会受等待、排期、代码评审等因素影响。

## 8. 依赖关系模型

依赖模型把两个任务拼成一个 pair 文本：

```text
SOURCE TASK: source_title + source_description
TARGET TASK: target_title + target_description
```

再做：

```text
pair text -> TF-IDF -> two-layer MLP -> relation label
```

关系标签来自 `task_links.tsv`，并被归一化成：

```text
duplicate
blocks
depends
related
clone
hierarchy
causal
supersedes
other_relation
no_relation
```

## 9. 为什么需要负样本

`task_links.tsv` 只包含“有关系”的任务对。如果只用这些训练，模型会以为任意两个任务都有关系。

所以训练时从同项目中随机抽取没有链接的任务对，标成：

```text
no_relation
```

这让模型能回答：

```text
这两个任务是否真的有关联？
```

## 10. 第一版的局限

第一版是轻量原型，优点是跑得快、可解释、易调试。局限是：

- TF-IDF 不理解深层语义，只看词和短语模式。
- 当前任务模型是三个独立 MLP，不是真正共享隐藏层。
- 依赖模型第一版使用拼接文本，暂时没有 GNN。
- TAWOS 是英文软件工程数据，中文任务需要后续中文微调。

## 11. 下一步升级路线

第一阶段：

```text
TF-IDF + 两层 MLP
```

第二阶段：

```text
SentenceTransformer embedding + 两层 MLP
```

第三阶段：

```text
BERT / DeBERTa 微调
```

第四阶段：

```text
GNN 处理任务图，预测缺失依赖边
```
