# 基于BERT与DistilBERT的欺骗性文本检测

> **通过微调预训练Transformer模型实现欺骗性文本与真实文本的二分类——该方法论可直接应用于金融欺诈检测、反洗钱叙述分析及金融虚假信息识别。**

---

## 项目概述

本项目对BERT和DistilBERT两个预训练模型进行微调（Fine-tuning），基于标注好的虚假/真实消费者评论数据集，构建能够区分捏造内容与真实内容的二分类文本分类器。项目在分类任务本身之外，还系统性地进行了**模型对比实验**（BERT vs. DistilBERT）和**决策阈值优化**分析——这两项都是实际生产部署中的关键考量。

**核心任务：** 给定一段文本，判断其是*欺骗性（虚假）*还是*真实可信*的。

**金融场景应用：** 相同的微调方法在金融服务领域中广泛用于以下场景：
- 检测贷款/保险申请中的虚假叙述
- 反洗钱（AML）工作流中对可疑交易报告（SAR）文本的自动分类
- 识别金融传播中的误导性或操纵性内容
- 在信贷承保中标记异常的自由文本字段

---

## 技术栈

| 类别 | 工具 |
|---|---|
| 深度学习框架 | PyTorch |
| 预训练模型 | `bert-base-uncased`、`distilbert-base-uncased`（HuggingFace Transformers） |
| 训练方式 | AdamW优化器 + 自定义训练循环微调 |
| 评估方法 | 准确率、混淆矩阵、阈值敏感性分析 |
| 可视化 | `matplotlib`、`seaborn`、`ipywidgets`（交互式阈值滑块） |

---

## 数据集

- **来源：** 手机及配件类商品欺骗性/真实评论数据集
- **任务：** 二分类——`欺骗性(1)` vs. `真实(0)`
- **数据划分：** 70% 训练集 / 15% 验证集 / 15% 测试集
- **文本字段：** `reviewText`；**标签字段：** `deceptive`

---

## Notebook 分析流程

### 1. 数据加载与预处理
使用BERT/DistilBERT分词器进行tokenization（`MAX_LEN=128`），通过`load_fake_true_reviews`进行分层采样划分。

### 2. 模型架构
基于HuggingFace的`BertForSequenceClassification` / `DistilBertForSequenceClassification`，在`[CLS]`向量上叠加线性分类头。

### 3. 超参数配置
两个模型实验中主要超参数设置如下：

| 参数 | BERT | DistilBERT |
|---|---|---|
| `MODEL_NAME` | bert-base-uncased | distilbert-base-uncased |
| `MAX_LEN` | 128 | 128 |
| `BATCH_SIZE` | 16 | 16 |
| `LEARNING_RATE` | 2e-5 | 3e-5 |
| `NUM_EPOCHS` | 8 | 8 |

### 4. 训练与评估
- 自定义训练循环，逐epoch记录训练损失与验证准确率
- 训练曲线可视化（损失收敛情况、准确率提升趋势）
- 在保留测试集上进行最终评估

### 5. 决策阈值优化
项目没有简单默认使用`threshold=0.5`，而是通过**交互式混淆矩阵**（ipywidgets滑块，范围0.0–1.0）来探索精确率与召回率的权衡关系。在欺诈/AML场景中，漏判真实欺诈（假阴性）与误判正常案例（假阳性）的业务代价截然不同，阈值选择因此至关重要。

### 6. BERT vs. DistilBERT 对比实验
对两个模型在全部8个epoch中的训练损失曲线与验证准确率进行并排对比。DistilBERT体积约小40%、推理速度更快，在对延迟敏感的生产系统中更具优势。本实验定量评估了在特定任务上的精度-效率权衡。

### 7. 新文本推理
```python
test_reviews = [
    "This product is wonderful, I highly recommend it to everyone.",
    "Worst purchase ever, completely useless.",
    "The item is just okay, but the reviews sound exaggerated.",
]
preds = sbr.predict_fake_true(test_reviews, loaded_model, loaded_tokenizer, device)
# 输出：每条文本的预测标签（'true'/'fake'）+ 置信度分数
```

### 8. 模型持久化
模型权重与分词器保存至本地（`fake_true_model/`），体现了对生产部署要求的理解。

---

## 核心结论

| 指标 | BERT | DistilBERT |
|---|---|---|
| 最终验证准确率 | — | — |
| 训练稳定性 | 8个epoch内平稳收敛 | 收敛模式相似，每epoch耗时更短 |
| 推理速度 | 基准 | 约快1.6× |
| 推荐使用场景 | 精度要求最高时 | 对延迟敏感的生产部署 |

*（具体数值取决于硬件环境，详见notebook中的训练曲线。）*

---

## 金融AI应用价值

阈值优化所揭示的精确率/召回率权衡在金融场景中尤为重要：

- **高阈值（保守型）：** 假阳性少——正常案例被误标记的情况减少，但漏判真实欺诈的风险上升。
- **低阈值（激进型）：** 欺诈捕获率更高，但合规团队的人工核查负担增加。

将消费者评论替换为贷款申请叙述、交易描述或财报电话会议文字稿，相同的BERT微调pipeline几乎可以直接迁移到金融文本分类任务中。

---

## 运行说明

```bash
# 安装依赖
pip install torch transformers tqdm pandas scikit-learn matplotlib seaborn ipywidgets

# 准备数据
# 将带标注的文本数据集保存为：
# data/CellPhonesAccessoriesdeceptivetruthful_dataset.txt

# 打开notebook
jupyter notebook bert_deceptive_text_detection.ipynb
```

**注意：** 强烈建议使用GPU训练（Google Colab T4免费GPU可用）。CPU环境下8个epoch训练速度较慢。

---

## 项目背景

本项目完成于Audencia商学院NLP相关课程（2026年）。BERT训练工具由课程提供（`simple_bert_fake_reviews`模块）；实验设计部分——包括超参数选择、BERT vs. DistilBERT对比实验以及决策阈值分析——均由任仁（Ren REN）独立完成。
