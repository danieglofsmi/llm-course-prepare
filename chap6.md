# 第六章 Finetuning for Classification
预训练之后，通常会对通用模型进行微调，可分为针对任务的微调和针对能力的指令微调。
第六章专注于分类任务的微调，任务更为具体、易于评估（Task-Specific Fine-Tuning）。通过这种方式，原本具有通用能力的大模型可以精确地预测特定标签空间内的任务，例如进行垃圾邮件检测、情感分析或主题分类。

## 代码
- 01_main-chapter-code 如何将一个仅经过预测下一个词（Next-token Prediction）训练的生成式解码器（Decoder-only）架构模型（gpt-2），转化为一个判别式分类器。系统性地演示了从原始数据集下载到使用gpt2训练，以及模型评估的全过程，讲解了如何通过保留预训练模型的大部分权重，给模型添加一个分类头实现分类，包括权重的冻结与解冻、损失函数的修改，在较小的标注数据集上实现知识的迁移。

- 02bonus_additional-experiments/additional-experiments.py：支持通过命令行参数调整微调的各个参数细节，包括层数、模型规模、训练token位置选择、上下文长度以及填充策略
  
- 03_bonus_imdb-classification：引入了 IMDb 电影评论数据集（50,000条样本） 。这一部分不仅扩展了实验的规模，还通过对比实验引入了 BERT、RoBERTa 以及更新的 ModernBERT 等编码器（Encoder）模型，从而在更高维度上探讨了生成式架构与判别式架构在分类任务中的优劣。


### 经典习题：
1. 在已有代码的基础上：
   1. 比较不同的模型上下文长度带来的效果差异
   2. 比较仅微调最后一个transformer block和微调整个模型参数的效果差异
   3. 比较微调第一个token和最后一个token的效果差异
2. kaggle：
   1. Mushroom Classification：Safe to eat or deadly poison?（https://www.kaggle.com/datasets/uciml/mushroom-classification）
   2. LLM Classification Finetuning：Finetune LLMs to Predict Human Preference using Chatbot Arena conversations（https://www.kaggle.com/competitions/llm-classification-finetuning/overview）


### 知识点
和其他章节的关系：
- 承接第五章预训练任务，开始后训练，由一个具体任务的微调引出第六章的指令微调
#### 核心目标：
    将预训练大语言模型（LLM）适配特定任务（分类）
#### 主要方法：
1. 预训练模型选型与适配：比较前面章节中的encoder和decoder在分类任务中的使用：
- Encoder 型模型：代表是 BERT（及其变体，如 DistilBERT）。双向注意力机制（无因果掩码），能够充分捕获文本中上下文的双向依赖关系。适合理解类任务，分类时通常取 [CLS] 特殊 token 的隐藏状态作为文本全局表征，接入分类头即可完成微调。
- Encoder-Decoder（/ 纯 Decoder）型模型：代表是本章实现的 GPT（纯 Decoder 模型，也可扩展为 T5 等 Encoder-Decoder 模型）。单向因果注意力机制（仅能关注当前 token 及之前的 token），适合生成类任务。需通过微调适配分类任务，通常取序列最后一个 token或第一个 token的隐藏状态作为文本表征，接入分类头。

2. 分词器与输入处理：统一文本预处理标准（分词、截断 / 填充至固定长度），确保输入格式与预训练阶段一致，避免破坏模型学到的语义规律。

3. 特征提取与分类头设计：
- 特殊 token 法：Encoder 模型用[CLS] token 特征，Decoder 模型用 EOS或 BOStoken 特征，直接作为文本全局表征。
- 分类头构建：在模型输出层后添加轻量型分类头（通常为 1-2 层全连接层），将高维特征映射到类别数维度（二分类输出 1 维，多分类输出 N 维），激活函数选用 Sigmoid（二分类）或 Softmax（多分类）。

4. 微调训练策略：
- 全参数微调：更新模型所有层参数，适用于数据集较大（如 10 万 + 样本）的场景，需较大计算资源。
- 部分参数微调（冻结策略）：冻结底层 Transformer 层（保留预训练的通用语义特征），仅更新顶层几层 Transformer 和分类头，降低过拟合风险与计算成本。
- 参数高效的微调策略（lora）
- 学习率设计：采用小学习率（通常 1e-5~1e-4），远低于预训练阶段，避免大幅破坏预训练权重；分类头可单独设置稍高学习率（如 1e-3），加速适配分类任务。

5. 优化器与损失函数：
- 优化器：优先使用 AdamW（带权重衰减），抑制过拟合；动量参数（β1、β2）沿用预训练经验值，保证训练稳定性。
- 损失函数：二分类用交叉熵损失（CrossEntropyLoss）或二元交叉熵损失（BCEWithLogitsLoss），多分类用交叉熵损失，确保损失计算与类别标签分布匹配。


## 论文/博客/视频
- 配套视频（https://www.youtube.com/watch?v=5PFXJYme4ik）（https://www.bilibili.com/video/BV1TConYWEiq/?spm_id_from=333.337.search-card.all.click&vd_source=f7fcafd1f7510675be530cdb12572d7b）
- 博客：Theory behind SFT（https://aiengineering.academy/LLM/TheoryBehindFinetuning/SFT/）
- 论文：How to Fine-Tune BERT for Text Classification?（https://arxiv.org/pdf/1905.05583）
- 技术报告：The Ultimate Guide to Fine-Tuning LLMs from Basics to Breakthroughs: An Exhaustive Review of Technologies, Research, Best Practices, Applied Research Challenges and Opportunities（chapter 6： Stage 4: Selection of Fine-Tuning Techniques and Appropriate Model Configurations）（https://arxiv.org/pdf/2408.13296）
- 拓展视频：2025生成式人工智能（吴恩达）ep13：Fine-tuning on a single task（防止灾难性遗忘）（https://www.bilibili.com/video/BV1sMEyzhEM3?spm_id_from=333.788.videopod.episodes&vd_source=f7fcafd1f7510675be530cdb12572d7b&p=13）
