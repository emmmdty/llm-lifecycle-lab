# LLM 全生命周期学习项目计划书

## 1. 项目目标

在当前 5090 服务器的单卡 RTX 5090 条件下，完整学习：

```text
公开数据
→ 数据检查与切分
→ Tokenizer
→ 小模型从零预训练
→ Continued Pretraining
→ SFT
→ Reward Model
→ DPO
→ GRPO
→ 多模态微调
→ 统一评测
→ 量化
→ vLLM / llama.cpp 部署
→ 质量与性能回归
```

这不是业务应用项目，也不要求训练出高质量通用模型。

项目的最终目标是能够解释并复现每个阶段，而不是堆积数据、模型或训练时长。

## 2. 核心决策：文本主线，多模态扩展

不要一步到位把全部流程建立在多模态模型上。

### 文本主线

文本模型用于学习最核心的机制：

- BPE/Unigram tokenizer；
- causal language modeling；
- label shift 和 causal mask；
- packing；
- optimizer、scheduler 和 gradient accumulation；
- CPT；
- assistant-only loss；
- LoRA/QLoRA；
- Reward Model；
- DPO；
- GRPO；
- 量化和部署。

### 多模态扩展

完成文本主线后，再学习：

- AutoProcessor；
- 图像 resize 和 pixel values；
- image token；
- multimodal collator；
- 视觉编码器冻结；
- VLM LoRA；
- 图文推理和部署。

多模态阶段不阻塞文本主线。

## 3. 模型路线

### 3.1 从零预训练模型

自行定义一个约 30M–60M 参数的 Decoder-only Transformer：

```yaml
vocab_size: 16000 或 32000
hidden_size: 512
num_hidden_layers: 8 到 12
num_attention_heads: 8
num_key_value_heads: 4
intermediate_size: 1536 或 2048
max_position_embeddings: 1024 或 2048
dtype: bfloat16
```

用途是学习完整预训练流程，不需要下载外部 checkpoint。

### 3.2 文本基础模型

主线模型：

```text
Qwen/Qwen3-0.6B-Base
```

用于：

- CPT；
- LoRA/QLoRA SFT；
- Reward Model；
- DPO；
- GRPO；
- 量化；
- vLLM 部署。

官方后训练对照：

```text
Qwen/Qwen3-0.6B
```

### 3.3 多模态模型

基础模型：

```text
Qwen/Qwen3.5-0.8B-Base
```

官方后训练对照：

```text
Qwen/Qwen3.5-0.8B
```

第一次服务器审计必须确认这些模型是否已下载完整，不得重复下载。

## 4. 数据路线

当前主线只使用规模可控、许可证可记录、可做有界抽样的数据集。任何大型网页语料都不能直接进入第一轮主线，必须先单独完成许可证、大小、split、schema 和磁盘预算审计。

### 4.1 从零预训练

主数据：

```text
modelscope/wikitext
```

特点：

- 约 522.66MB；
- 来自精选 Wikipedia 文章；
- 规模适合小模型预训练；
- 文本自然，比纯合成语料更适合观察 validation loss。

快速闭环数据：

```text
AI-ModelScope/TinyStories
```

用途：

- 先训练 5M–20M 参数模型；
- 快速验证 tokenizer、packing、loss、checkpoint 和生成；
- 小模型容易观察到连贯文本生成。

推荐安排：

```text
TinyStories：只做快速预训练闭环
Wikitext：作为 30M–60M 模型的正式教学语料
```

### 4.2 中文 CPT

使用小型中文领域文本：

```text
AI-ModelScope/tigerbot-law-plugin
```

已公开信息：

- 约 29.87MB；
- 55,895 行；
- Apache-2.0；
- 包含 title、chapter、content 等字段。

从其中提取正文，构造 300万–800万 token 的领域 CPT 语料。

选择法律文本不是为了训练法律模型，而是为了形成明显的“通用英语预训练 → 中文领域分布适配”对照。

若不希望使用法律语料，可让 Codex在 ModelScope 中查找一个：

- 小于 2GB；
- 明确许可证；
- 纯文本；
- 不需要全量网页抓取；
- 至少数百万 token；

的中文技术或金融领域语料替换。替换前必须记录理由和许可证。

### 4.3 中文 SFT

使用：

```text
AI-ModelScope/alpaca-gpt4-data-zh
```

只抽取 5,000–20,000 条完成主实验。

目标是学习：

- instruction/input/output 转 messages；
- chat template；
- assistant-only loss；
- Full、LoRA、QLoRA 的区别。

### 4.4 Reward Model 和 DPO

使用：

```text
llamafactory/ultrafeedback_binarized
```

只抽取：

```text
Reward Model：5,000 对
DPO：8,000 对
Validation：1,000 对
```

必须按 prompt 分组切分，不能让同一个 prompt 的偏好对进入不同 split。

### 4.5 GRPO

主数据改为：

```text
AI-ModelScope/gsm8k
```

已公开信息：

- 约 5.90MB；
- MIT；
- 小学数学文字题；
- 答案可以程序化解析。

只使用 500–2,000 道训练题即可。

OpenR1-Math-220k 改为高级可选项，不进入第一轮主线。

### 4.6 多模态

使用：

```text
lmms-lab/ChartQA
```

只抽取 2,000–5,000 条训练样本，并保留独立验证/测试子集。

用途是学习：

- 图像读取；
- processor；
- VLM message 格式；
- 图像 token；
- 多模态 LoRA；
- 图表问答评测。

## 5. 数据总预算

建议服务器最终保留：

| 数据 | 预算 |
|---|---:|
| TinyStories | 原始数据集或不超过 2GB |
| Wikitext | 约 0.7GB 磁盘 |
| 中文 CPT 文本 | 不超过 1GB |
| 中文 SFT | 不超过 1GB |
| UltraFeedback 子集 | 不超过 2GB |
| GSM8K | 小于 20MB |
| ChartQA | 以实际盘点为准，训练子集 2K–5K |
| 处理后 Parquet | 总计尽量不超过 10GB |

这个项目不需要 TB 级数据。

## 6. GPU 时间预算

| 阶段 | 建议 GPU |
|---|---:|
| 环境 smoke | 10–20 分钟 |
| TinyStories 快速预训练 | 0.5–2 小时 |
| Wikitext 30M–60M 预训练 | 2–6 小时 |
| CPT | 1–3 小时 |
| SFT | 1–4 小时 |
| Reward Model | 1–3 小时 |
| DPO | 1–4 小时 |
| GRPO | 1–5 小时 |
| VLM-SFT | 2–6 小时 |
| 评测、量化、部署 | 每项 1–4 小时 |

每次正式运行默认不超过 8 小时，硬上限 10 小时。

## 7. 最终产物

项目完成后应具备：

1. 数据审计和许可证清单；
2. 自建 tokenizer；
3. 小模型从零预训练 checkpoint；
4. CPT adapter；
5. SFT adapter；
6. Reward Model；
7. DPO adapter；
8. GRPO 运行结果；
9. 可选 VLM adapter；
10. 统一评测报告；
11. BF16、NF4、W4A16、GGUF 结果；
12. vLLM 和 llama.cpp 性能报告；
13. 每个阶段的命令、配置、指标和失败分析。
