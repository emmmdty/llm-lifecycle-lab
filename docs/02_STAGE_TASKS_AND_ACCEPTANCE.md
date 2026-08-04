# 阶段任务与执行验收

## 通用规则

每个阶段必须依次完成：

```text
阅读原理
→ 检查输入
→ 编写最小代码
→ 单元测试
→ CPU 或单 batch 测试
→ 最多 5 step GPU smoke
→ 100–200 step 性能基准
→ 人工决定是否正式运行
→ 验收报告
```

Codex 不得在 smoke test 后自动开始正式训练。

每次正式运行必须记录：

- Git commit；
- 完整命令；
- resolved config；
- Python、Torch、CUDA 和 GPU；
- 模型和数据 revision；
- seed；
- loss 和关键指标；
- tokens/s、step time 和峰值显存；
- checkpoint；
- 失败与限制。

---

## 阶段 0：项目和服务器审计

### 任务

- 检查本地项目已有代码和文档；
- 连接服务器；
- 检查 GPU、驱动、CUDA、磁盘、uv、Git 和 ModelScope；
- 盘点已有 `.venv-*`、`models/` 和 `data/`；
- 判断模型/数据是完整、部分还是损坏；
- 检查服务器现有 GitHub 代理镜像配置。

### 验收

- 输出事实清单和未确认事项；
- 不修改系统配置；
- 不下载任何资源；
- 不启动 GPU；
- 明确已有模型和数据的目录及大小；
- 给出下一阶段可执行条件。

---

## 阶段 1：环境

### 环境划分

```text
.venv-train
.venv-eval
.venv-quant
.venv-serve
```

### 任务

- 使用 uv 创建或审计环境；
- 保存各环境 freeze；
- 验证 PyTorch CUDA 13；
- 验证 BF16 matmul；
- 验证 Transformers 模型加载；
- 验证 lm-eval；
- 验证 LLM Compressor import；
- 验证 vLLM 独立环境。

### 验收

- 每个环境 `uv pip check` 通过；
- 不出现旧 vLLM 源码编译；
- RTX 5090 compute capability 正确；
- BF16 无 NaN；
- lm-eval 完成 5 条样本 smoke；
- 环境可重建。

---

## 阶段 2：数据治理

### 任务

为所有数据实现统一流程：

```text
盘点
→ 许可证
→ schema 检查
→ 有界抽样
→ 清洗
→ 去重
→ 按 document/prompt 分组切分
→ Parquet
→ token 统计
→ manifest
```

### 验收

- 不包含未经审计的大型网页语料；
- 所有数据有 revision 和许可证；
- 没有无上限下载；
- 同一 prompt 不跨 split；
- test 不进入训练；
- 抽样索引和 seed 可复现；
- 处理后数据总量受控。

---

## 阶段 3：Tokenizer

### 任务

- 用 TinyStories 或 Wikitext 训练 16K/32K BPE；
- 固定 BOS、EOS、PAD、UNK；
- 比较自建 tokenizer 与 Qwen tokenizer；
- 分析英文、中文和代码的 tokens/character。

### 验收

- encode/decode 基本可逆；
- special token ID 固定；
- tokenizer 可保存和重新加载；
- 训练语料 revision 可追溯；
- 给出 vocabulary 对模型参数和序列长度的影响。

---

## 阶段 4：TinyStories 快速预训练

### 任务

先用 5M–20M 模型跑通：

- causal mask；
- label shift；
- packing；
- AdamW；
- scheduler；
- gradient accumulation；
- checkpoint/resume；
- validation loss；
- 文本生成。

### 验收

- loss 持续下降；
- checkpoint 恢复后曲线连续；
- 训练后生成比随机模型明显连贯；
- 记录 tokens/s 和显存；
- 总运行不超过 2 小时。

---

## 阶段 5：Wikitext 正式教学预训练

### 任务

训练约 30M–60M 模型：

- sequence length 1024；
- 训练 token 上限 3,000万–8,000万；
- BF16；
- cosine schedule；
- validation perplexity；
- checkpoint resume。

### 验收

- 无 NaN/Inf；
- validation loss 下降；
- resume 可用；
- 训练耗时不超过 8 小时；
- 能解释参数、activation、gradient 和 optimizer 显存。

---

## 阶段 6：Qwen3 CPT

### 任务

- 从 `Qwen3-0.6B-Base` 开始；
- 将中文领域正文转 causal LM 数据；
- 做 LoRA-CPT；
- 准备 domain held-out 和 general held-out；
- 比较 Base 与 CPT perplexity。

### 验收

- 使用原 Qwen tokenizer；
- domain perplexity 改善；
- 通用退化被量化；
- adapter 可重新加载；
- 训练不超过 3 小时。

---

## 阶段 7：SFT

### 任务

做两个实验：

1. 小模型 Full-SFT；
2. Qwen3 LoRA/QLoRA-SFT。

实现：

- instruction/input/output 转 messages；
- chat template；
- assistant-only loss；
- packing；
- adapter merge。

### 验收

- prompt token 不计算 assistant loss；
- held-out SFT loss 下降；
- 固定 50 条 prompt 前后对比；
- LoRA 合并前后输出一致；
- 比较 Full、LoRA、QLoRA 显存和速度。

---

## 阶段 8：Reward Model

### 任务

- 使用 chosen/rejected pair；
- 添加 scalar classification head；
- 实现 pairwise Bradley–Terry loss；
- 统计 preference accuracy 和 reward margin；
- 检查长度偏置。

### 验收

- preference accuracy 高于随机；
- 同 prompt 不跨 split；
- chosen/rejected reward 分布可视化；
- 报告 reward 与回答长度相关性；
- 保存可重新加载的模型。

---

## 阶段 9：DPO

### 任务

- 从 SFT checkpoint 开始；
- 明确 policy 和 reference；
- 记录 chosen/rejected log probability；
- 调整 beta、learning rate 和 epoch；
- 比较 SFT 与 DPO。

### 验收

- preference margin 提升；
- reference checkpoint 可追溯；
- 通用评测没有灾难性下降；
- 固定 prompt 对比；
- 训练不超过 4 小时。

---

## 阶段 10：GRPO

### 任务

- 使用 GSM8K；
- 每个 prompt 生成 2–4 个回答；
- 实现最终答案解析；
- 分离 exact-answer reward 与 format reward；
- 执行短 GRPO。

### 验收

- exact accuracy 和格式正确率分开报告；
- 平均 reward 不替代准确率；
- 人工检查至少 50 条 rollout；
- 不出现只优化格式的 reward hacking；
- 训练不超过 5 小时。

---

## 阶段 11：多模态扩展

### 任务

- 加载 Qwen3.5-0.8B-Base 和 processor；
- 处理 ChartQA 图像与 messages；
- 冻结视觉编码器起步；
- 为语言层/连接层添加 LoRA；
- 限制图像尺寸；
- 执行 2K–5K 样本短训练。

### 验收

- image token 未被截断；
- 单图、纯文本推理可用；
- ChartQA 子集比 Base 改善；
- 纯文本能力无明显退化；
- 记录预处理时间和峰值显存。

---

## 阶段 12：统一评测

### 任务

统一评测：

- held-out loss/perplexity；
- GSM8K；
- C-Eval 子集；
- HellaSwag；
- 固定中文指令；
- preference validation；
- ChartQA test 子集。

### 验收

- checkpoint、数据 revision 和 generation config 固定；
- 所有结果写入统一表；
- 不可比较项明确标记；
- 评测失败不静默忽略；
- 无训练—评测泄漏。

---

## 阶段 13：量化

### 任务

比较：

```text
BF16
NF4
W4A16
GGUF Q8_0
GGUF Q4_K_M
```

### 验收

- 所有产物能重新加载；
- 校准集不含 test；
- 比较磁盘、显存和质量；
- 记录转换命令和版本；
- 量化质量下降可解释。

---

## 阶段 14：部署

### 任务

- vLLM 部署 BF16/W4A16；
- llama.cpp 部署 GGUF；
- OpenAI-compatible API；
- 并发 1、4、8；
- 固定输入输出长度。

### 验收

记录：

- TTFT；
- TPOT；
- output tokens/s；
- requests/s；
- P50/P95；
- error rate；
- peak GPU memory；
- 同一质量评测。

---

## 阶段 15：最终验收

### 验收问题

学习者必须能解释：

- 数据为什么这样选择；
- 每个 loss 对什么优化；
- CPT、SFT、DPO、GRPO 的区别；
- Full、LoRA、QLoRA 的适用边界；
- checkpoint 如何恢复；
- 数据如何避免泄漏；
- 量化如何影响质量和性能；
- vLLM 与 llama.cpp 的定位；
- 共享 GPU 环境中如何安全运行。
