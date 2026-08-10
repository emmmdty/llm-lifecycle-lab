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

### 环节拆解（2026-08-10 升级：把每个环节拆细，缺项视为未完成）

| 环节 | 内容 | 产出物 |
| --- | --- | --- |
| 1 原理 | 写出该环节的原理（公式 / 流程图 / 与文献对照），明确学习目标 | 阶段报告"原理"节 |
| 2 输入盘点 | 数据 / 模型 / checkpoint 现状、许可证、revision、大小与预算 | manifest / 资产记录 |
| 3 最小实现 | 核心机制亲手实现（不抄现成库；与参考实现数值对照） | 源码 |
| 4 单元测试 | 本地 synthetic fixture 覆盖核心逻辑 | pytest 通过 |
| 5 CPU / 单 batch | 服务器 CPU 或单 batch 跑通 | 日志 |
| 6 GPU smoke | 最多 5 step | 日志 + run.json |
| 7 性能基准 | 100–200 step，含 resume 连续性，据此估算正式时长 | 基准日志 + 报告 |
| 8 正式运行 | 人工确认后；有 max_steps；预算内 | runs/ 完整记录 |
| 9 验收 | 逐项对照验收标准，每条给证据 | 验收报告 |
| 10 教程 | 撰写该阶段教程（真实问题与解决过程） | docs/tutorials/NN_*.md |

每阶段记录必须包含：Git commit、完整命令、resolved config、Python/Torch/CUDA/GPU、模型和数据 revision、seed、loss 与关键指标、tokens/s、step time、峰值显存、checkpoint、失败与限制。

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

- 先决策 Q2（docs/06）：Wikitext-103 约 1.03 亿 token 与 30M–60M 模型存在数据-规模冲突（tokens/param 仅 1.7–3.4，严重欠训练）。选项：降模型规模匹配 D≈20N、维持规模并量化记录欠训练、或扩语料（违反小语料原则，不推荐）。决策理由与证据必须写入阶段 5 报告并回写 docs/06；
- 执行 Q1 多规模对比实验（正式任务，2026-08-05 用户决策列入）：在 TinyStories 上训练至少 3 个规模（例如 5M / 18M / 64M，**最小规模用小参数模型节省时间**）各 1 epoch，记录 val loss，拟合 val loss vs tokens/param 曲线，验证 D≈20N 在本项目语料上的适用性；结论回写 docs/06；
- 验证 docs/06 登记册 Q3/Q7/Q8/Q10/Q12/Q13（多 epoch 收益、序列长度/位置编码、评价升级、MFU 纳入记录、词表占比、深窄 vs 宽浅）；
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
- 单次大显存 GPU 任务默认 ≤8 小时，可放宽，硬上限 <24 小时（放宽需在运行记录中说明理由）；
- 能解释参数、activation、gradient 和 optimizer 显存；
- Q2 决策已记录（docs/06 回写）；
- Q1 多规模对比实验给出 ≥3 个规模的数据点与 val loss 结论（docs/06 回写）。

### 完成状态（2026-08-05）

阶段 5 已完成：Q1（5.14M/18.1M/65.4M 三规模，α=0.124、R²=0.998，D≈20N 框架适用）、Q2 决策（正式 5.32M 匹配 D≈20N + 26.8M 欠训练对照）、Wikitext 80M token 正式预训练（5.32M val ppl 114.37 / 26.8M val ppl 43.32，2441 步各 2–4 分钟）、登记册 Q3/Q8/Q10/Q12 验证。全部验收项证据见 docs/00「阶段 5」补充；详细结论见 docs/06 第 2 节与登记册。

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

### 完成状态（2026-08-05）

阶段 6 已完成：tigerbot-law 领域 LoRA-CPT（Qwen3-0.6B-Base，rank-2 adapter 573K 可训练参数，6ND 决策见 docs/06 Q15），domain held-out（title 分组 3.91M token）+ general held-out（wikitext/tinystories validation，Qwen tokenizer 编码），Base vs CPT 对比评测，Q8/Q9 顺带验证。全部验收项证据见 docs/00「阶段 6」补充、reports/cpt-compare-formal.json、runs/20260805-211848。

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

### 完成状态（2026-08-06）

阶段 7 已完成：SFT 数据准备（中文 alpaca-gpt4-zh + 英文 alpaca-cleaned 均按 prompt 分组切分，token+assistant-mask 流 + manifest）、三实验（tiny Full-SFT / Qwen3 LoRA r8 / Qwen3 QLoRA r8 NF4）、assistant-only loss、packing、checkpoint/resume、merge 一致性、Full vs LoRA vs QLoRA 资源对比。全部验收项证据见 docs/00「阶段 7」补充、reports/stage7-eval-summary.json、runs/20260806-082602（tiny）/ 20260806-082619（LoRA）/ 20260806-083036（QLoRA）；Q16–Q20 登记见 docs/06；真实问题（QLoRA NF4 merge 有损、Qwen3 空 think 块、小模型生成退化）记录在教程 05。

---

## 阶段 8：主线从零预训练（2026-08-10 新增）

背景：项目定位升级（docs/01 §1.1、§3.1）——主线教学模型规模按"尽可能大"原则结合实测数据决策：minimind_dataset 治理后实测 ~1.40B tokens（Qwen3 口径）→ 用户确认**严格 D≈20N，N≈70M**。原小模型资产（TinyStories 18.1M、Wikitext 5.32M/26.8M）保留为教程对比项。

### 任务（按环节拆解）

1. **原理**：复习 D≈20N 缩放规律与 Q1/Q2 验证结论（docs/06 第 2 节）；写出本阶段的规模决策公式 `N = D/20` 与时间校验 `6·N·D / 实测吞吐 ≤ 时间预算`。
2. **主线语料治理**（Q26）：
   - 候选：minimind_dataset pretrain（ModelScope，Apache-2.0，1.2GB mini / 10GB 主线）；
   - 核对许可证、revision、大小、schema、去重；有界抽样；按行/document 分组切分 held-out；token 流（沿用阶段 2 流程）；manifest；
   - 记录与 minimind 原始用法（直接自回归文本流）的差异：我们做治理、切分、manifest，不照搬；
   - 实测治理后 token 总数，决定规模区间。
3. **规模决策**（Q21）：用户已确认严格 D≈20N（N=D/20≈70M，2026-08-10）；治理后按选定 tokenizer 实测修正（预期 60M–85M 区间），数据受限选项（128M–200M）未采用并记录理由；校验 `6·N·D / 实测吞吐 ≤ 8h`（~70M 无需放宽）。
4. **训练**：复用 src/pretrain 框架（dry-run / smoke / bench / resume / run.json / metrics.jsonl），单卡 5090、BF16、cosine、seq 1024、max_steps 上限；200M 级约 40K–80K steps，需评估 checkpoint 保留策略（≤16 个）。
5. **对照评测**（教程对比项）：与 TinyStories 18.1M、Wikitext 5.32M/26.8M、Q1 65.4M 数据点对照 val loss/ppl（同口径）；生成质量检查（固定 prompt、distinct 4-gram）。
6. **教程**：撰写主线预训练教程（含与 minimind 同规模级对比的定位说明、规模决策全过程）。

### 验收

- 数据治理产物齐全（manifest、token 流、held-out 切分、与 minimind 原始用法差异记录）；
- 规模决策有完整计算证据（D≈20N + 时间预算 + 实测吞吐）；
- 无 NaN/Inf；validation loss 下降；resume 连续性验证通过；
- 单次任务 ≤8h（128M 方案）或放宽记录（200M 方案，<24h 硬上限）；
- 与既有小模型数据点有对照表；
- checkpoint 可重新加载；生成样本入库；教程完成。

---

## 阶段 9：主线 SFT（2026-08-10 新增）

背景：阶段 7 已教过 SFT 全流程（tiny Full-SFT + Qwen3 LoRA/QLoRA）。本阶段把 SFT 应用到主线模型，教学点是**容量与数据的过渡带**（18M 退化 → 64M+ 正常 → 0.6B 强基座的三档容量对照）。

### 任务（按环节拆解）

1. **原理**：assistant-only loss、chat template、Full vs LoRA/QLoRA 的资源权衡（复习 docs/06 Q16–Q20）。
2. **数据**：中文 alpaca-gpt4-zh（语言匹配决策 Q16 的应用：主线模型预训练语料为中文，SFT 用中文数据；英文 alpaca-cleaned 留作英文能力对照）。
3. **训练**：主线模型 Full-SFT（~70M 显存量级可承受，参照阶段 7 tiny Full 3.59GB）；packing、mask 流沿用阶段 7。
4. **对照**：三档容量对照表（18.1M 退化 [Q19] / 主线模型 / Qwen3-0.6B LoRA），固定 50 prompt 前后对比。
5. **生成验证**：固定 prompt 人工检查；distinct 指标；退化诊断（如发生，记录原因）。
6. **教程**：更新 SFT 教程或撰写主线 SFT 章节。

### 验收

- assistant-only loss 下降；
- 三档容量对照表（含与 Q19 的对比结论）；
- 固定 50 prompt 前后对比报告；
- checkpoint 可重新加载；
- 训练 ≤4 小时。

---

## 阶段 10：Reward Model

载体：主线模型 Full-RM（scalar head 直接加在主线模型上）为主，Qwen3-0.6B LoRA-RM 为强基座对照；教学点包括小容量 RM 的 preference accuracy 与长度偏置。

### 任务（按环节拆解）

1. **原理**：Bradley–Terry pairwise loss、scalar head、preference accuracy 口径。
2. **数据**：ultrafeedback_binarized，按 prompt 分组切分（5,000 对 train / 1,000 对 validation），同一 prompt 不跨 split；治理 test 不进入训练。
3. **实现**：chosen/rejected 同 prompt 配对、标量 head、pairwise loss、reward margin 统计。
4. **训练**：dry-run → smoke → bench → 正式（1–3h 预算）；主线 Full-RM 与 Qwen3 LoRA-RM 双实验。
5. **分析**：chosen/rejected reward 分布可视化；reward 与回答长度相关性（长度偏置）；对抗样本（重复、拒答、格式）。
6. **教程**：RM 教程（含与 DPO 的衔接说明）。

### 验收

- preference accuracy 高于随机；
- 同 prompt 不跨 split；
- chosen/rejected reward 分布可视化；
- 报告 reward 与回答长度相关性；
- 保存可重新加载的模型（主线 + Qwen3 两条线）。

---

## 阶段 11：DPO

载体：policy = 主线 SFT（阶段 9 产物），reference = 主线 SFT 冻结副本；Qwen3 线（阶段 7 SFT 产物）为对照。

### 任务（按环节拆解）

1. **原理**：DPO 的隐式 reward、KL 约束、beta 的作用；与 RM（阶段 10）的衔接与差异。
2. **数据**：ultrafeedback_binarized 8,000 对（prompt 分组，与 RM 阶段同一治理产物）。
3. **实现**：policy/reference 双模型、chosen/rejected log-prob、DPO loss、无需在线 rollout。
4. **训练**：从 SFT checkpoint 开始；调整 beta、learning rate、epoch；主线 + Qwen3 对照。
5. **评估**：preference margin 提升；KL/相对偏移；通用评测不灾难性下降；固定 prompt 对比。
6. **教程**：DPO 教程。

### 验收

- preference margin 提升；
- reference checkpoint 可追溯；
- 通用评测没有灾难性下降；
- 固定 prompt 对比；
- 训练不超过 4 小时。

---

## 阶段 12：GRPO

载体：主线模型（生成质量受限则如实记录）为主，Qwen3 为对照。

### 任务（按环节拆解）

1. **原理**：在线策略梯度、组内 advantage 归一化、reward 与 advantage 的关系。
2. **数据**：GSM8K（500–2,000 题），答案程序化解析。
3. **实现**：policy 生成 2–4 个回答/ prompt；exact-answer reward 与 format reward 分离。
4. **训练**：短 GRPO；干跑 → smoke → bench → 正式（≤5h）。
5. **评估**：exact accuracy 与格式正确率分开报告；平均 reward 不替代准确率；人工检查至少 50 条 rollout；防 reward hacking。
6. **教程**：GRPO 教程。

### 验收

- exact accuracy 和格式正确率分开报告；
- 平均 reward 不替代准确率；
- 人工检查至少 50 条 rollout；
- 不出现只优化格式的 reward hacking；
- 训练不超过 5 小时。

---

## 阶段 13：多模态扩展

载体：Qwen3.5-0.8B-Base（文本主线不阻塞，独立扩展线）。

### 任务（按环节拆解）

1. **原理**：processor、image token、vision encoder + projector 结构。
2. **数据**：ChartQA（2K–5K 训练样本 + 独立验证/测试子集）。
3. **实现**：messages 含图像；冻结视觉编码器起步；语言层/连接层 LoRA；图像尺寸限制。
4. **训练**：短训练；记录预处理时间和峰值显存。
5. **评估**：image token 未被截断；单图、纯文本推理可用；ChartQA 子集比 Base 改善；纯文本能力无明显退化。
6. **教程**：多模态教程。

### 验收

- image token 未被截断；
- 单图、纯文本推理可用；
- ChartQA 子集比 Base 改善；
- 纯文本能力无明显退化；
- 记录预处理时间和峰值显存。

---

## 阶段 14：统一评测

对象：主线模型全量（预训练 → SFT → DPO/GRPO 各节点）+ Qwen3 对照线。

### 任务（按环节拆解）

1. **清单**：held-out loss/perplexity、GSM8K、C-Eval 子集、HellaSwag、固定中文指令、preference validation、ChartQA test 子集。
2. **实现**：统一评测脚本；checkpoint、数据 revision、generation config 固定。
3. **执行**：全部结果写入统一表；不可比较项明确标记。
4. **分析**：链路各节点的能力曲线（预训练 → SFT → RL 的逐阶段变化）；与 docs/06 评价清单对照。
5. **教程**：统一评测教程。

### 验收

- checkpoint、数据 revision 和 generation config 固定；
- 所有结果写入统一表；
- 不可比较项明确标记；
- 评测失败不静默忽略；
- 无训练—评测泄漏。

---

## 阶段 15：量化

对象：主线模型（BF16/NF4/W4A16/GGUF），Qwen3 可选。

### 任务（按环节拆解）

1. **原理**：NF4 分位量化、W4A16 离线校准量化、GGUF 量化格式差异。
2. **实现**：LLM Compressor（.venv-quant）与 llama.cpp 转换；校准集不含 test。
3. **执行**：比较磁盘、显存和质量（同一 tokenizer、prompt、generation config、评测集）。
4. **记录**：转换命令、工具版本、量化质量下降解释。
5. **教程**：量化教程。

### 验收

- 所有产物能重新加载；
- 校准集不含 test；
- 比较磁盘、显存和质量；
- 记录转换命令和版本；
- 量化质量下降可解释。

---

## 阶段 16：部署

对象：主线模型（vLLM BF16/W4A16、llama.cpp GGUF），Qwen3 可选。

### 任务（按环节拆解）

1. **实现**：vLLM 部署（OpenAI-compatible API）；llama.cpp GGUF 部署。
2. **性能基准**：并发 1、4、8；固定输入输出长度。
3. **记录**：TTFT、TPOT、output tokens/s、requests/s、P50/P95、error rate、peak GPU memory。
4. **质量回归**：同一质量评测（阶段 14 口径）。
5. **教程**：部署教程。

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

## 阶段 17：最终验收

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
- 共享 GPU 环境中如何安全运行；
- 主线模型（~70M，D≈20N 严格匹配）的规模决策过程（实测数据 → D≈20N → 时间预算校验）；
- 小模型资产（18.1M/5.32M/26.8M）作为对比项的教学结论（容量-能力曲线）；
