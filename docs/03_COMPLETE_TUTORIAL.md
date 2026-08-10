# LLM 从数据到部署完整教程

## 第一部分：先建立正确的学习目标

真实基础模型往往使用数十亿到数万亿 token，但学习训练链路不需要复制这种规模。

本项目通过"小数据、小模型、完整过程"学习：

1. 数据如何转换为 token；
2. 模型如何通过 next-token prediction 学习；
3. 预训练模型如何适应新领域；
4. 模型如何学习指令格式；
5. 偏好数据如何影响回答风格；
6. 在线奖励如何更新策略；
7. 权重如何量化；
8. 模型如何成为服务。

小实验的价值在于每个环节都可观察、可失败、可修复。

**主线模型（2026-08-10 定位升级）**：本项目的主线教学模型规模按"尽可能大"原则与实测数据结合决策——minimind_dataset 实测 ~1.40B tokens（Qwen3 口径），用户确认严格 D≈20N → **~70M**（规模决策方法见 docs/01 §3.1 与 docs/06 Q21），贯穿后续 SFT、RM、DPO、GRPO、统一评测、量化、部署全链路；TinyStories 18.1M、Wikitext 5.32M/26.8M 等小模型资产保留为教程中的规模/容量对比项；Qwen3-0.6B 为强基座对照线。

---

## 第二部分：数据

### 1. 原始数据与处理后数据

目录概念：

```text
data/raw        原始下载，不修改
data/processed  清洗、切分、物化后的数据
data/manifests  来源、revision、许可证和统计
```

不要让训练代码直接读取来源不明的散乱文件。

### 2. Manifest 必须记录

```yaml
dataset_id:
revision:
license:
download_date:
source_files:
original_rows:
selected_rows:
estimated_tokens:
text_fields:
filters:
dedup_method:
split_strategy:
seed:
checksums:
```

### 3. 切分顺序

正确：

```text
按 document_id 或 prompt hash 分组
→ 切分 train/validation/test
→ 再展开 chosen/rejected/rollouts
```

错误：

```text
先展开偏好回答
→ 随机按行切分
```

错误方式会让相同 prompt 出现在训练和验证中。

---

## 第三部分：Tokenizer

Tokenizer 把字符串映射为整数序列。

### 需要学习

- normalization；
- pre-tokenization；
- BPE merge；
- vocabulary；
- special tokens；
- encode/decode；
- token compression。

### 实验

训练 16K 和 32K 两个 tokenizer，比较：

- 参数量；
- 平均序列长度；
- 中文、英文、代码 tokens/character；
- OOV/UNK；
- validation loss。

Vocabulary 增大会增加 embedding 和 LM head 参数，但可能减少序列长度。

---

## 第四部分：从零预训练

### 1. 输入和标签

输入：

```text
[BOS, t1, t2, t3]
```

目标：

```text
[t1, t2, t3, EOS]
```

模型学习：

```math
L = - Σ log p(x_t | x_<t)
```

### 2. 必须理解的实现

- causal attention mask；
- label shift；
- padding mask；
- sequence packing；
- batch 和 gradient accumulation；
- BF16；
- AdamW；
- warmup 和 cosine decay；
- gradient clipping；
- checkpoint；
- validation perplexity。

### 3. 三步预训练（2026-08-10 升级）

先用 TinyStories 验证实现（阶段 4，已完成），再用 Wikitext 做小模型正式实验（阶段 5，已完成），最后在治理后的主线语料（minimind_dataset，阶段 8）上训练 ~70M 级主线教学模型（D≈20N 严格匹配）。

TinyStories 的结果应容易观察：

- 随机模型输出混乱；
- 短训练后出现简单故事结构；
- loss 和生成同步改善。

Wikitext 更接近自然百科文本，用于更真实的 validation loss 和 perplexity 学习（5.32M/26.8M，教程对比项）。

主线预训练（阶段 8）的教学点：数据治理与规模决策方法（D≈20N × 时间预算，见 docs/06 Q21），以及与 minimind 同规模级对比的定位说明。

---

## 第五部分：Continued Pretraining

CPT 仍然使用 causal LM loss，但数据分布变了。

```text
Qwen3 Base
+ 中文领域纯文本
→ LoRA-CPT
```

需要同时准备：

- domain validation；
- general validation。

若 domain loss 降低但 general loss 大幅上升，说明发生遗忘。

CPT 不能使用指令模板，也不应把问答数据直接当成对话 SFT。

---

## 第六部分：SFT

SFT 让模型学习：

- 用户和助手角色；
- 指令遵循；
- 回答格式；
- 目标回答风格。

### 消息格式

```json
{
  "messages": [
    {"role": "user", "content": "问题"},
    {"role": "assistant", "content": "回答"}
  ]
}
```

### Assistant-only loss

通常只对 assistant 内容计算 loss：

```text
system/user tokens → label = -100
assistant tokens   → 正常 label
```

必须检查 chat template 是否支持正确的 generation mask。

### Full、LoRA、QLoRA

- Full：更新全部权重，显存最大；
- LoRA：冻结基础权重，训练低秩矩阵；
- QLoRA：基础权重 4-bit 加载，LoRA 仍使用较高精度。

小模型 Full-SFT 用于学习完整优化；Qwen3 使用 LoRA/QLoRA。

---

## 第七部分：Reward Model

Reward Model 输入 prompt + answer，输出标量。

偏好目标：

```text
reward(chosen) > reward(rejected)
```

典型 pairwise loss：

```math
L = -log σ(r_chosen - r_rejected)
```

不要只看 loss，还要看：

- preference accuracy；
- reward margin；
- reward 分布；
- 长度偏置；
- 重复、拒答、格式等对抗样本。

---

## 第八部分：DPO

DPO 不需要在线 rollout，也不需要在训练过程中直接调用 Reward Model。

它比较 policy 对 chosen/rejected 的相对概率，同时以 reference policy 约束偏移。

关键超参数：

- beta；
- learning rate；
- epochs；
- max prompt/completion length。

评估必须同时看：

- preference margin；
- KL 或相对偏移；
- 通用任务；
- 固定 prompt 人工对照。

---

## 第九部分：GRPO

GRPO 是在线训练流程：

```text
prompt
→ policy 生成多个回答
→ 奖励函数评分
→ 同组标准化 advantage
→ 更新 policy
```

GSM8K 适合第一次实验，因为答案可验证。

奖励建议分开：

```text
exact_answer_reward
format_reward
```

如果只看总 reward，模型可能通过输出格式获得高分，却没有提高答案准确率。

---

## 第十部分：多模态

VLM 比文本模型多出：

```text
图片
→ processor
→ pixel values / image tokens
→ vision encoder
→ projector / unified hidden states
→ language decoder
```

需要防止：

- 图像分辨率导致 batch 显存波动；
- 文本截断删除 image token；
- processor 与模型不匹配；
- LoRA 目标模块硬编码错误；
- 数据 collator 只适用于单图。

第一次实验冻结视觉编码器，只训练语言层和连接层 LoRA。

---

## 第十一部分：评测

### 训练指标

- train loss；
- validation loss；
- perplexity；
- grad norm；
- tokens/s；
- step time；
- peak GPU memory。

### 能力指标

- GSM8K exact match；
- C-Eval accuracy；
- HellaSwag accuracy；
- preference accuracy；
- ChartQA accuracy；
- 固定 prompt 人工评分。

一个阶段的指标改善不能证明模型“全面变强”。

---

## 第十二部分：量化

### NF4

主要用于 QLoRA 和运行时 4-bit 加载。

### W4A16

权重 INT4，activation BF16/FP16。需要校准数据并输出离线量化模型。

### GGUF

面向 llama.cpp：

- Q8_0：质量高、文件较大；
- Q4_K_M：更小、更常用。

量化比较必须使用相同 tokenizer、prompt、generation config 和评测集。

---

## 第十三部分：部署

### vLLM

适合：

- GPU 服务；
- continuous batching；
- OpenAI-compatible API；
- 多并发吞吐。

### llama.cpp

适合：

- GGUF；
- CPU/GPU 混合；
- 单机轻量部署。

### 性能指标

- TTFT：第一个 token 等待时间；
- TPOT：后续每个 token 时间；
- tokens/s；
- requests/s；
- P50/P95；
- error rate；
- peak memory。

---

## 第十四部分：实验复现

每次运行都要保存：

```text
run_id
command
resolved_config
git commit
environment
hardware
model revision
dataset revision
seed
metrics
logs
checkpoint
```

没有这些信息，即使训练成功，也不能称为可复现。

---

## 第十五部分：使用 Codex 的正确方式

不要说：

```text
一次性完成整个 LLM 全链路项目
```

应拆成：

```text
实现阶段 2 的数据 manifest 和切分
只使用 synthetic fixture
先写测试
不下载数据
不启动 GPU
```

完成后独立审查，再进入下一阶段。

每次让 Codex回答：

1. 当前阶段的输入是什么？
2. 代码范围是什么？
3. 如何验证？
4. 哪些命令需要在服务器运行？
5. GPU 运行上限是什么？
6. 哪些结论仍未确认？
