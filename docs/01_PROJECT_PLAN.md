# LLM 全生命周期学习项目计划书

## 1. 项目目标

在当前 5090 服务器的单卡 RTX 5090 条件下，完整学习：

```text
公开数据
→ 数据检查与切分
→ Tokenizer
→ 从零预训练（主线模型 100M–200M 级，2026-08-10 升级）
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

### 1.1 项目定位：与经典项目的差异（2026-08-10 更新）

本项目是 **LLM 全链路教学项目**：以 5090 单卡可负担的最大规模（100M–200M 级主线模型），把数据治理 → Tokenizer → 从零预训练 → CPT → SFT → Reward Model → DPO → GRPO → 多模态 → 统一评测 → 量化 → 部署全链路的**每个环节拆细**：原理 → 最小实现 → 单元测试 → smoke → 基准 → 正式运行 → 验收 → 教程。

与两个经典开源项目的关系（**不是重复造轮子，差异在"过程证据"**）：

- **MiniMind（jingyaogong/minimind）**：产品/复现项目。目标是用最低成本训练出可用的 64M 级模型，价值是"结果"（模型与廉价训练配方），不提供数据治理、缩放律验证、统一评测、量化/部署基准、失败分析等过程证据。本项目与其同规模级训练存在的意义：用严格的阶段验收把 minimind 只展示"最终指标"的环节变成可解释、可复现、可对照的教学过程。
- **动手学 LLM（datawhalechina/happy-llm）**：工具使用教程。概念讲解好，但实现层大量调用 ms-swift / llama-factory 等现成框架，学习者动手的是配置与命令，不是机制本身。本项目每个环节亲手实现核心机制（自建 tokenizer、自写训练循环、自写 RM head / Bradley–Terry / DPO / GRPO loss），用单元测试、合成数据、服务器实测证据验收；教程每章带本项目实测数字与失败案例。

**自我约束（防止"为完成项目而完成项目"）**：

1. 训练循环、loss、量化等代码与 TRL / llama-factory / LLM Compressor 功能重叠——存在理由是"教学 + 自测"，必须保留"与参考实现数值对照"的验证（阶段 7 已有 TRL import 对照先例），不抄代码、不把现成库的调用当作实现；
2. 教程结构可与 happy-llm 章节重合，但每章必须包含本项目实测数据、失败案例与验收证据；
3. 主线预训练与 minimind 同规模级——验收重点不是"再训一个 64M"，而是数据治理、规模决策方法（D≈20N × 时间预算）、缩放对照与全链路后续环节（Q25）。

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

**主线教学模型（2026-08-10 升级）**：5090 单卡可承载的**尽可能大**的 Decoder-only Transformer。规模不是固定值（64M 只是示例），由三个约束在阶段 8 执行时决策：

1. **数据预算（主约束，2026-08-10 实测后更新）**：主线预训练语料治理后实测 token 数——minimind_dataset pretrain_t2t.jsonl（8.27GB，8,468,827 行）按 Qwen3 tokenizer 抽样外推 **~1.40B tokens**（chars/token=1.65，见 data/manifests/minimind-dataset.json）。D≈20N 下 1.4B tokens ≈ **70M 模型**；200M 需要 4B tokens，当前数据不可达（补充语料会占用硬盘，待用户决策，Q21）。
2. **时间预算**：单次任务默认 ≤8h，可放宽，硬上限 <24h。按实测吞吐 ~66 TFLOPS（docs/06 §4.3）外推：8h 内 compute-optimal 约 128M（2.6B token）；24h 上限约 200M（4B token）。>250M 超出 24h 硬上限，不在本轮范围。
3. **显存（非约束）**：32GB 显存实测可训练 ~1B 级模型，200M 级峰值约 6GB。

**决策公式**（沿用 Q1/Q2 已验证框架）：`N = D/20`，再校验 `6·N·D / 实测吞吐 ≤ 时间预算`。

候选区间（2026-08-10 实测后）：严格 D≈20N 匹配为 **~70M**（t/p≈20）；接受数据受限（minimind 先例 t/p≈7.8）可上 **128M–200M**（t/p≈7–11，欠训练量化记录）。最终规模在阶段 8 数据治理（选定 tokenizer 实测 token 数）与 150 步基准后确定并记录（Q21）。

候选配置起点（阶段 5 Q1 已验证 65.4M 的放大版，阶段 8 执行时决策）：

```yaml
vocab_size: 32768（32K 词表；Q12 已实测 65.4M 级 embedding 占比 ~13%，可接受；词表精简对比登记 Q22）
hidden_size: 768–1024
num_hidden_layers: 12–16
num_attention_heads: 12–16
intermediate_size: 3072–4096
max_position_embeddings: 1024
dtype: bfloat16
```

用途：主线教学模型，后续环节（主线 SFT、RM、DPO、GRPO、统一评测、量化、部署）全部以其为载体，Qwen3-0.6B 作为强基座对照。

**历史小模型资产（保留为教程对比项，不再是主线载体）**：TinyStories 18.1M（阶段 4）、Wikitext 5.32M / 26.8M（阶段 5，Q2 决策）——作为规模/容量/欠训练对照数据点写入主线预训练教程。

### 3.2 文本基础模型（对照基座线，2026-08-10 定位更新）

强基座对照线：

```text
Qwen/Qwen3-0.6B-Base
```

用于：

- CPT（已完成，阶段 6）；
- LoRA/QLoRA SFT（已完成，阶段 7）；
- Reward Model / DPO / GRPO 的强基座对照（阶段 10–12，主线模型为主、Qwen3 为对照）；
- 量化和部署对照（阶段 15–16）。

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

### 4.1 主线从零预训练（阶段 8，2026-08-10 新增）

主数据候选：

```text
gongjy/minimind_dataset（ModelScope，2026-08-10 已下载至 data/raw/minimind_dataset）
```

已核实事实（2026-08-10 实测，manifest 见 data/manifests/minimind-dataset.json）：

- 许可证：ModelScope yaml 标注 **CC-BY-NC-4.0**（HF 标注 Apache-2.0，冲突以更严格者记录）；
- `pretrain_t2t.jsonl` 实测 **8.27GB**、8,468,827 行、sha256 校验通过；`pretrain_t2t_mini.jsonl` 1.24GB、1,270,238 行；
- schema 统一 `{"text": "..."}`，zh + en（中文为主）；
- Qwen3 tokenizer 抽样外推主文件 **~1.40B tokens**（chars/token=1.65）；
- 样本含大量指令/对话风格文本，质量分布需在治理时检查；
- mini 与主文件是否重叠需在治理时核对去重；
- minimind 原始用法是直接当自回归文本流（max_seq_len≈380/768），本项目必须按阶段 2 流程重新治理（许可证核对、去重、有界抽样、按行/document 分组切分 held-out、token 流、manifest）。

规模与数据预算由 D≈20N 决策框架在阶段 8 确定（Q21）：实测 ~1.4B tokens 下严格匹配为 ~70M，数据受限可上 128M–200M（t/p≈7–11）。硬盘预算：raw ~9.5GB + int32 token 流 ≤6GB + checkpoint 保留 ≤16 个 ≈ 合计 ≤20GB。

### 4.2 从零预训练（历史教学数据，保留为教程对比项）

主数据：

```text
modelscope/wikitext
```

特点：

- 约 522.66MB；
- 来自精选 Wikipedia 文章；
- 规模适合小模型预训练（阶段 5 已完成）；
- 文本自然，比纯合成语料更适合观察 validation loss。

快速闭环数据：

```text
AI-ModelScope/TinyStories
```

用途：

- 先训练 5M–20M 参数模型（阶段 4 已完成）；
- 快速验证 tokenizer、packing、loss、checkpoint 和生成；
- 小模型容易观察到连贯文本生成。

推荐安排（历史记录，2026-08-10 更新：主线语料见 4.1）：

```text
TinyStories：快速预训练闭环（已完成）
Wikitext：5.32M/26.8M 教学预训练（已完成，教程对比项）
```

### 4.3 中文 CPT

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

### 4.4 中文 SFT

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

2026-08-06 补充：中文 alpaca-gpt4-zh 用于 Qwen3 LoRA/QLoRA-SFT（语言匹配）；另下载英文 alpaca-cleaned（cc-by-4.0，44MB）用于小模型 Full-SFT（小模型为英文预训练，语言匹配决策见 docs/06 Q16）。两者均按 user prompt 分组切分防跨 split 泄漏，治理 test 不进入训练。

### 4.5 Reward Model 和 DPO

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

### 4.6 GRPO

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

### 4.7 多模态

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
| 主线预训练（minimind_dataset，阶段 8） | raw ~9.5GB + int32 token 流 ≤6GB |
| 中文 CPT 文本 | 不超过 1GB |
| 中文 SFT | 不超过 1GB |
| UltraFeedback 子集 | 不超过 2GB |
| GSM8K | 小于 20MB |
| ChartQA | 以实际盘点为准，训练子集 2K–5K |
| 处理后 Parquet | 总计尽量不超过 10GB |
| 主线 checkpoint | 保留 ≤16 个（200M 级每个约 0.4GB） |

主线预训练资产（raw + token 流 + checkpoint）合计控制在 **~20GB 以内**（2026-08-10 用户决策：硬盘为约束，不占用过多）。

这个项目不需要 TB 级数据。

## 6. GPU 时间预算

| 阶段 | 建议 GPU |
|---|---:|
| 环境 smoke | 10–20 分钟 |
| TinyStories 快速预训练 | 0.5–2 小时 |
| Wikitext 小模型预训练 | 2–6 小时 |
| CPT | 1–3 小时 |
| SFT | 1–4 小时 |
| **主线预训练（100M–200M，阶段 8）** | **2–20 小时（~200M 需按放宽规则记录理由；128M 目标 8h 内）** |
| **主线 SFT（阶段 9）** | **1–4 小时** |
| Reward Model | 1–3 小时 |
| DPO | 1–4 小时 |
| GRPO | 1–5 小时 |
| VLM-SFT | 2–6 小时 |
| 评测、量化、部署 | 每项 1–4 小时 |

每次正式运行默认不超过 8 小时；时间可放宽，但小于 1 天（24 小时）是硬上限。放宽需在运行记录中说明理由。

## 7. 最终产物

项目完成后应具备：

1. 数据审计和许可证清单；
2. 自建 tokenizer；
3. **主线模型从零预训练 checkpoint（100M–200M 级，阶段 8；小模型 checkpoint 作为教程对比项保留）**；
4. CPT adapter；
5. **主线 SFT checkpoint（阶段 9）+ Qwen3 LoRA/QLoRA 对照**；
6. **Reward Model（主线模型为主 + Qwen3 对照）**；
7. **DPO adapter（同 6）**；
8. **GRPO 运行结果（同 6）**；
9. 可选 VLM adapter；
10. **统一评测报告（主线模型 + Qwen3 对照线）**；
11. **BF16、NF4、W4A16、GGUF 结果（主线模型为主）**；
12. **vLLM 和 llama.cpp 性能报告（主线模型为主）**；
13. 每个阶段的命令、配置、指标和失败分析。
