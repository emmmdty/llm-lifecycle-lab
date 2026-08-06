# 开放问题与缩放规律验证计划（跨阶段登记册）

> 本文档登记与"模型规模、数据规模、训练效果评价"相关的开放问题。
> 原则：**只记录，不顺手实现**。每个问题标注建议验证阶段，由对应阶段任务在执行时逐项验证，并把结论回写到本文档（更新"状态"与"结论"）。

## 1. 项目的最终目标是什么？

项目使命（docs/01）：不是训练一个"最好的模型"，而是能够**解释并复现每个阶段**。因此"我们最终训练多大的模型"本身不是答案，答案是一套**可解释的决策方法**：

```text
给定语料规模 D 和 GPU 预算
→ 用缩放规律估计最优参数规模 N
→ 设定可验证的训练目标（val loss/ppl、生成质量、任务级评测）
→ 训练、评价、记录，与文献/基线对比
```

## 2. 数据规模与模型规模的匹配（Chinchilla 缩放规律）

**当前理解**：Hoffmann et al. (2022) 给出 compute-optimal 训练下 D ≈ 20N（token/参数 ≈ 20）。模型偏大或数据偏少都会欠训练。

| 场景 | tokens/param | 判断 |
| --- | ---: | --- |
| 阶段 4 实际：18.1M / TinyStories 392M | **21.7** | ≈ 最优 ✓ |
| 假想：64M / TinyStories 392M | 6.1 | 欠训练（数据不足） |
| 阶段 5 计划：30M / Wikitext 103M | 3.4 | 严重欠训练 ✗ |
| 阶段 5 计划：60M / Wikitext 103M | 1.7 | 严重欠训练 ✗ |
| Wikitext 103M 的最优规模 | ~5.2M | 数据允许的规模上限 |

**待验证问题 Q1**（阶段 5 前，可选基准）：在 TinyStories 上各跑 1 epoch（64M 估算 15–30 分钟）对比 18M/32M/64M 的 val loss，验证 D≈20N 在本项目语料上的适用性。

**Q1 验证结论（✅ 已验证，2026-08-05 阶段 5 完成）**：在 TinyStories 392M token（16k 词表）上训练 5.14M / 18.1M / 65.4M 各 1 epoch：

| 规模 | tokens/param | val loss | ppl | 拟合 loss |
| --- | ---: | ---: | ---: | ---: |
| 5.14M | 76.3 | 1.971 | 7.18 | 1.9625 |
| 18.1M（阶段 4） | 21.7 | 1.665 | 5.28 | 1.6794 |
| 65.4M | 6.0 | 1.439 | 4.22 | 1.4329 |

拟合 `L(N) = 13.26 · N^(-0.124)`，R² = 0.998；指数 α=0.124 落在 Chinchilla/GPT-3 文献 N 指数区间（~0.1）。结论：**本项目管线的 loss–规模幂律成立，D≈20N 的决策框架适用**；固定数据下 18M→64M 边际收益递减（Δ=0.226 < 5M→18M 的 0.306），符合数据受限区特征。注意：单语料实验只能验证 N 维幂律；D 维指数需跨语料/子集实验（未做，记录于 Q14 备注）。

**待决策 Q2**（阶段 5 正式任务）：阶段 5 的"30M–60M 模型 + Wikitext 103M"存在数据-规模冲突。选项：
1. 降低模型规模至 5–15M，用全量 Wikitext，匹配 Chinchilla；
2. 维持 30M–60M，接受欠训练并**量化记录**（val 曲线、与文献对比、多 epoch 收益）；
3. 扩大语料（违反"小语料"原则，需重新走数据治理，不推荐）。
决策需在阶段 5 记录理由并更新 docs/01、docs/06。

**Q2 决策（✅ 已决策，2026-08-05）**：选择**选项 1（降模型规模匹配 D≈20N）**——正式教学模型 5.32M（32k 词表 embedding 下限，h=128/5 层，seq 1024），80M token 预算下 tokens/param=15.0；同时按用户要求补跑 **26.8M（tokens/param=3.0）作为欠训练对照**。实测（Wikitext，32k 词表，实际流 92.31M token，训练 79.99M token）：

| 模型 | tokens/param | 算力(6ND) | val loss | ppl |
| --- | ---: | ---: | ---: | ---: |
| 5.32M（正式） | 15.0 | 2.5e15 | 4.7394 | 114.37 |
| 26.76M（对照） | 3.0 | 12.9e15 | 3.7685 | 43.32 |

解读（重要，避免误读）：26.8M 绝对 loss 更低是因为投入了 5× 算力，不是"D≈20N 失效"。按 Chinchilla 固定算力最优分配，26.8M 对应最优规模应约 10M（若数据允许扩到 ~200M token）；其 t/p=3.0 处于**数据受限区**——继续加参不如加数据。5.32M 在 80M 数据下同时接近数据最优与算力最优。决策已同步 docs/01（模型路线 §3.1）。

**Q2 延伸（✅ 已验证，2026-08-05 阶段 6）：D≈20N 决策方法应用于 LoRA 可训练参数量（Q15）**。tigerbot-law domain_train 经 Qwen3 tokenizer 实测 D=3,912,160 token：

- unique 读法（Chinchilla 的 D 指语料规模）：N_target = D/20 ≈ 196K；
- seen 读法（训练 3 epoch，FLOPs≈6ND_seen，D_seen=3D=11.74M）：N_target = 3D/20 ≈ 587K；
- 候选（q/k/v/o，Qwen3 实际维度 q/o 为 2048 维）：rank 1=286,720（unique 1.47×）、rank 2=573,440（seen **0.98×**、unique 2.93×）、rank 4=1,146,880；
- **决策：rank 2（573,440 可训练参数，base 596M 的 0.096%）**——匹配 seen 读法；相对 unique 读法放宽 2.9×，理由：base 已预训练，adapter 只学领域分布偏移（与"从零训练"的 N 语义不同），且 rank≥2 保证低秩更新稳定性；
- 实测（360 步/3 epoch/seed 42）：domain_val ppl 7.015→4.795（-31.6%），无通用退化（见 docs/00 阶段 6 补充）。

## 3. 我们的真实瓶颈在哪里？

阶段 4 实测：峰值显存 1.22GB / 32GB（3.8%）、606K tokens/s、MFU ≈ 17%（实测有效算力 ~66 TFLOPS / 5090 FP16 峰值约 380 TFLOPS）。

2026-08-05 服务器能力实测检查（扩大训练规模的可行性）：

| 资源 | 实测 | 结论 |
| --- | --- | --- |
| 磁盘 | 3.6T 总量，2.0T 空闲，inode 233M 用 1% | 可容纳 100GB 级语料与 checkpoint ✓ |
| 显存 | 单卡 32GB（当前占用 4.6GB，空闲充足） | BF16 可训练 ~1B 参数模型 ✓ |
| 网络 | ModelScope 可达（HTTP 200）；**HuggingFace 不可达** | 大数据只能经 ModelScope 下载 |
| 时间 | 规则（2026-08-05 更新）：单次大显存任务默认 ≤8h，可放宽，硬上限 **<24h** | 单卡 5090 可训练约 24h×606K t/s ≈ 52B tokens/任务 |
| 数据可得性 | 候选：minimind_dataset（ModelScope，Apache-2.0，pretrain 1.2GB~10GB 级） | 64M 级训练数据可下载 ✓ |

结论：**物理资源允许显著扩大训练规模**（数据可下载、磁盘/显存/算力都够），真正的约束是项目"小语料"原则与每阶段时间预算（规则层）。是否调整属于计划决策（Q11）。

按 8h 默认预算 + 实测吞吐推算，单卡 5090 每个任务 compute-optimal（D=20N）的上限约 **875M 参数 / 17.5B tokens**；放宽到 24h 上限时约 **2.6B 参数 / 52B tokens**。64M 规模（minimind-3 级别，~5 亿 token）只需约 50 分钟。

**瓶颈排序 = 数据量 > 时间/教学预算 > 显存 > 算力**。
- 数据：TinyStories 392M、Wikitext 103M，全项目数据总量受控（docs/04 预算）；
- 时间：阶段预算 2–8h，单卡；
- 显存：任何 ≤500M 参数的模型在 32GB 上都绰绰有余；
- 算力：18M 全语料仅 10.8 分钟；64M 估算 15–30 分钟。minimind 等开源项目在 3090 上 2h 训练 64M 级模型，佐证算力非瓶颈。

## 4. 社区经验登记（以小见大的参照系）

### 4.1 minimind-3（jingyaogong/minimind，Apache-2.0，2026-04 主线）

README 事实（2026-08-05 核实）：

- 模型：64M 参数，词表 **6400**、dim 768、8 层、q_heads 8 / kv_heads 4、RoPE、max_pos 32768；MoE 变体 198M-A64M（4 experts / top-1）；
- **"2 小时"的准确含义是 SFT 阶段单卡 3090 跑 1 epoch**（README 明确标注），不是端到端预训练；Zero 路线（pretrain_t2t_mini + sft_t2t_mini）合计约 2.31h；
- 预训练数据：`pretrain_t2t_mini.jsonl` 1.2GB（快速复现，max_seq_len≈768）/ `pretrain_t2t.jsonl` 10GB（主线，max_seq_len≈380）；中文约 1.5~1.7 字符/token；
- 训练开销（单卡 3090）：64M pretrain_t2t_mini ≈ 1.21h ≈ 1.57 元；全部阶段合计约 3 元；
- 经验点：小模型词表宜精简（embedding/LM head 参数占比）；固定参数量下**深而窄优于宽而浅**（引用 MobileLLM：125M/350M 时 30~42 层优于 ~12 层；d_model<512 劣势放大，>1536 时加深更划算）；跨 tokenizer 比较建议用 BPB 而非 PPL。

### 4.2 公开模型规模对照（社区数据）

| 模型 | 参数 | 训练数据 | tokens/param | 训练算力 |
| --- | --- | --- | ---: | --- |
| 本项目阶段 4 | 18.1M | 392M | 21.7 | 4.3e16 FLOPs，单卡 5090 10.8 分钟 |
| minimind-3 | 64M | ~5 亿（mini 集） | ~7.8 | 单卡 3090 ~1.2h（实测） |
| GPT-3 | 175B | 300B | 1.7 | 单卡 5090 折算 ~55,440 天 |
| Chinchilla | 70B | 1.4T | 20 | 论文：最优计算分配 D≈20N |
| LLaMA-2 | 7B / 70B | 2T | 286 / 29 | 2048×A100-80G 约 1 个月 |
| DeepSeek-V3 | 671B MoE（激活 37B） | 14.8T | 400（按激活） | 约 278 万 H800 GPU·小时 |

### 4.3 外推方法与结论

训练 FLOPs ≈ 6 × 参数 × token 数（每 token 每参数约 6 FLOPs，前向+反向）。以本项目实测点（65.8 TFLOPS 有效）为单卡基准外推：

- 64M / 5 亿 token：≈ 50 分钟（与 minimind 实测 3090 1.2h 互证，外推方法有效）；
- 1B / 20B token（D=20N 最优）：≈ 21 天单卡；
- 7B / 2T：≈ 14,784 天单卡（≈ 40 年）；
- 70B / 2T：≈ 148 万天单卡；DeepSeek-V3：≈ 578 万天单卡。

**为什么"多几个 epoch"不能替代更大规模**：固定语料 D 下，epoch 只是重复同一个 D，信息量不增加；参数量 N 与数据 D 需要同比例增长（D≈20N），FLOPs ≈ 6ND 因此随 N² 增长（N↑ 且 D≈20N↑）。万亿参数模型需要 10T 级以上数据 + 数十万 GPU·小时，这是数据采集与算力基础设施的壁垒，不是超参与训练技巧能跨越的——这正是"以小见大"要建立的尺度感：小模型实验的价值在于用可负担的成本验证方法链，并把实测点外推到真实大模型的规模。

## 4. 现有资源约束下如何获得更好的训练效果（按优先级）

**数据侧（收益最大）**
- Q3：多 epoch 收益——TinyStories 1 vs 2 vs 3 epoch 的 val loss/生成对比（每 epoch 仅 ~10 分钟，成本极低）。阶段 5 或阶段 4 收尾可选实验。**✅ 已验证（阶段 5，5.14M 模型）：1 epoch val loss 1.971 → 3 epochs 1.769（Δ=-0.202，ppl 7.18→5.87）**；关键对照：**18.1M×1 epoch（1.665）优于 5.14M×3 epochs（1.769）**——固定数据下加参数比加 epoch 更有效，与 D≈20N 预测一致。
- Q4：数据质量与噪声的影响（TinyStories 相对干净；如需可做子集对比）。

**训练侧**
- Q5：超参敏感性——lr 扫描（1e-4/3e-4/1e-3）、有效 batch 大小、warmup 比例、min_lr。
- Q6：正则——dropout 当前为 0（1 epoch 欠拟合场景合理）；多 epoch 场景需评估 dropout>0。
- Q7：序列长度与位置编码（512 learned-absolute vs 1024/RoPE）对生成质量的影响。**（阶段 5 未验证：512→1024 同时换了语料与词表，对照被混淆；留待阶段 12 或专门实验）**

**评价侧**
- Q8：效果评价升级——仅 val ppl 不够。需补充：val-train gap（过拟合诊断）、生成重复率/多样性、固定 prompt 人工对照（已有 samples.jsonl 基础）、任务级评测（阶段 12 统一评测；此前可引入轻量过渡评测）。**✅ 已验证（阶段 5 + 阶段 6）：val-train gap 已纳入 metrics（全部运行 gap>0 且小：0.017–0.147）；生成多样性（distinct 4-gram ratio）已纳入（TinyStories 1.0、Wikitext 0.59–0.63、阶段 6 CPT 0.632→0.655）；阶段 6 固定 3 prompt（法律/中文日常/英文）Base vs CPT 生成对照入 reports/cpt-compare-formal.json；任务级评测留待阶段 12。**

## 5. 如何评价模型训练效果（诊断清单）

| 层面 | 指标 | 状态 |
| --- | --- | --- |
| 训练曲线 | train/val loss、ppl、grad norm、lr | 已记录（metrics.jsonl） |
| 稳定性 | NaN/Inf、loss spike、seed 敏感性 | 已记录（无 NaN）；seed 敏感性 Q9 ✅（阶段 6 双 seed <0.1%）；resume 时 BF16 最后一位非确定性（~1e-5 相对误差） |
| 效率 | tokens/s、step time、峰值显存、**MFU** | 前三已记录；MFU 已纳入运行记录（Q10 ✅，阶段 5：5M=3.4%、18M≈17%、64M=16.6%、Wiki-5.3M=4.1%、Wiki-26.8M=9.3%，基准 5090 FP16 dense 380 TFLOPS，config 字段 `peak_flops`） |
| 基线对比 | 随机初始化 loss（≈ln V=9.70，实测 9.81）、训练后下降幅度 | 已有 |
| 拟合诊断 | val-train gap、多 checkpoint val 曲线 | 已有（Q8 ✅：全部运行 gap>0 且小） |
| 能力 | 任务级评测、生成人工对照 | 阶段 12 + 生成样本（已有，Q8 ✅ 生成多样性量化） |

## 6. 训练更高参数量级的突破路径（记录，分阶段引入）

| 突破方向 | 内容 | 计划阶段 |
| --- | --- | --- |
| 数据 | 更大语料（治理、许可证、去重） | 各阶段按需 |
| 并行 | DDP/FSDP、多卡教学实验 | 阶段 5+ 明确需要时 |
| 注意力效率 | FlashAttention/SDPA（已用）、窗口/稀疏注意力、KV cache | 生成/部署阶段 |
| 显存 | gradient checkpointing、8-bit/Adafactor 优化器 | 模型显著变大时 |
| 利用率 | 更大 batch、序列打包（已用）、kernel 优化、MFU 追踪 | 阶段 5 |
| 训练稳定性 | loss spike 恢复、EMA、数据顺序（随机块 vs 顺序）对比 | 阶段 5+ |

## 7. 登记册状态

| 编号 | 问题 | 验证阶段 | 状态 |
| --- | --- | --- | --- |
| Q1 | TinyStories 上多规模（5M/18M/64M，最小规模用小参数模型省时）各 1 epoch 对比 val loss，验证 D≈20N；2026-08-05 用户决策正式列入阶段 5 任务 | 阶段 5（正式任务） | ✅ 已验证（α=0.124，R²=0.998，幂律成立；详见第 2 节） |
| Q2 | 阶段 5 模型规模与 Wikitext 语料的匹配决策 | 阶段 5 | ✅ 已决策（2026-08-05：正式 5.32M 匹配 D≈20N + 26.8M 欠训练对照；详见第 2 节） |
| Q3 | 多 epoch 收益（1 vs 2 vs 3） | 阶段 5 | ✅ 已验证（5.14M：1 ep 1.971 → 3 ep 1.769；18M×1ep 仍优于 5M×3ep） |
| Q4 | 数据质量/噪声影响 | 阶段 5+ | 待验证 |
| Q5 | 超参敏感性（lr/batch/warmup/min_lr） | 阶段 5+ | 待验证 |
| Q6 | dropout 在多 epoch 场景的作用 | 阶段 5+ | 待验证 |
| Q7 | 序列长度与位置编码对比 | 阶段 5 | 未验证（512→1024 换语料/词表，对照混淆；留待阶段 12 或专门实验） |
| Q8 | 效果评价升级（gap/多样性/过渡评测） | 阶段 5/12 | ✅ 已验证（阶段 5：val-train gap + 4-gram 多样性纳入；阶段 6 补充：固定 3 prompt Base vs CPT 生成对照 + 多样性 0.632→0.655；任务级评测留待阶段 12） |
| Q9 | seed 敏感性（同配置多 seed 曲线差异） | 阶段 5+ | ✅ 已验证（阶段 6：CPT 全配置 360 步 seed 42 vs 43，domain ppl 4.801 vs 4.799 <0.1%、general <1%；resume 时 BF16 最后一位非确定性依旧存在） |
| Q10 | MFU 纳入运行记录 | 阶段 5 | ✅ 已验证（metrics/run.json 均有 mfu；config 增加 peak_flops 字段） |
| Q11 | 扩大训练规模的正式决策：**2026-08-05 已决策**——物理资源允许（磁盘 2TB、显存 32GB、ModelScope 可达、64M 级约 50 分钟），修改规则为"单次大显存 GPU 任务默认 ≤8h，可放宽，硬上限 <24h"（AGENTS.md/docs/01/docs/05 已同步）；扩大语料（如 minimind_dataset，Apache-2.0）需走数据治理流程 | 决策已完成 | ✅ 已决策 |
| Q12 | 词表大小与参数占比（minimind 6400 vs 本项目 16384；embedding 占比对总参数量影响） | 阶段 5 | ✅ 已验证（embedding 占比：TS-5M=42%、TS-18M=48%、TS-64M=13%、Wiki-5.3M=**81%**、Wiki-26.8M=65%；32k 词表在小模型上 embedding 严重挤占参数，佐证 minimind 精简词表经验） |
| Q13 | 深窄 vs 宽浅（MobileLLM；本项目 hidden=512 处于 d_model<512 劣势边界） | 阶段 5 | 未验证（5M 为深窄 h=128×15 层、18M/64M 为 h=512，规模不同不可直接对照；留待专门实验） |
| Q14 | 跨 tokenizer 效果评价改用 BPB（Bits Per Byte） | 阶段 5/12 | 待验证（阶段 5 记录了相关事实：32k BPE 编码 Wikitext 得 92.3M token vs manifest 16k 估算 80M，byte-level BPE 对 wiki 标记文本的压缩率低于普通英文） |
| Q15 | D≈20N 决策方法扩展到 LoRA-CPT 可训练参数量 | 阶段 6 | ✅ 已验证（rank 2=573,440 匹配 seen 读法 3D/20≈587K（0.98×）；unique 读法放宽 2.9×；详见第 2 节 Q2 延伸） |
| Q16 | 小模型 SFT 数据语言匹配（英文 vs 中文） | 阶段 7 | ✅ 已决策并验证（小模型用英文 alpaca-cleaned，Qwen3 用中文 alpaca-gpt4-zh；详见第 2 节 Q16） |
| Q17 | QLoRA NF4 在 Blackwell sm_120 上的可行性与显存 | 阶段 7 | ✅ 已验证（bitsandbytes 0.50.0 NF4 在 RTX 5090 可用，峰值显存 7.43GB；详见第 2 节 Q17） |
| Q18 | QLoRA adapter merge 到 NF4 基座的一致性 | 阶段 7 | ✅ 已验证（merge 进 NF4 基座有损 ppl 6.65→8.88（PEFT 官方警告）；去量化到 BF16 后 merge 一致，loss 差 2.1e-4、生成 10/10；详见第 2 节 Q18） |
| Q19 | 小模型（18.1M）SFT 后的生成退化现象 | 阶段 7 | ✅ 已记录（held-out assistant loss -43.5% 但生成退化到 `*` 重复；18M 容量不足以学指令遵循，诚实记录；详见第 2 节 Q19） |
| Q20 | chat template 对 SFT 训练/评测口径的影响（Qwen3 空 think 块） | 阶段 7 | ✅ 已验证（训练时模板对最后一条 assistant 消息插空 `<think>\n\n</think>\n\n`；生成时需 enable_thinking=False 才能对齐，否则产生前缀噪声；详见第 2 节 Q20） |

## 8. 阶段 7 SFT 决策与验证记录

### Q16：小模型 SFT 数据语言匹配（✅ 2026-08-06 已决策并验证）

小模型（TinyStories/Wikitext 预训练，英文 BPE）与 Qwen3（中英双语）的 SFT 数据语言必须匹配：

| 实验 | SFT 数据 | 语言 | 理由 |
| --- | --- | --- | --- |
| tiny Full-SFT | alpaca-cleaned（英文，cc-by-4.0，44MB 新下载） | 英文 | 小模型 tokenizer 只见过英文，中文 token 化碎片化严重（实测 8 字→30 token vs 7 词→7 token） |
| Qwen3 LoRA/QLoRA | alpaca-gpt4-zh（中文，CC BY-NC 4.0，治理产物） | 中文 | Qwen3 预训练含中文，与阶段 6 CPT 领域一致 |

英文数据选择 alpaca-cleaned（yahma 清洗版）而非合成指令：真实指令数据、许可清晰（cc-by-4.0）、规模可控（44MB 单文件，治理后 12K/1.5K/1.5K）。合成指令方案记录为备选（数据真实性差，不推荐）。

### Q17：QLoRA NF4 on Blackwell sm_120（✅ 2026-08-06 已验证）

bitsandbytes 0.50.0 + transformers 5.14.1 在 RTX 5090（sm_120）上 NF4 4-bit 基座 + LoRA 训练可用：dry-run、5 step smoke、150 step bench、222 step 正式训练全部通过，峰值显存 7.43GB（vs LoRA BF16 基座 14.13GB，节省 ~47%）。性能代价：tokens/s 27.9K→19.0K（-32%），MFU 0.548→0.220（NF4 反量化开销）。

### Q18：QLoRA merge 到 NF4 基座有损（✅ 2026-08-06 已验证，PEFT 已知警告）

`merge_and_unload` 直接作用在 NF4 基座上会把 LoRA 权重再量化进 4-bit，产生精度损失：held-out assistant ppl 6.65→8.88（PEFT 源码警告 "may get different generations due to rounding errors"）。正确流程：**先加载 BF16 基座 → 挂 adapter → merge**，此时 loss 差 2.1e-4、生成 10/10 一致（与 BF16 LoRA 的 3.3e-4/5/5 同量级）。这是 QLoRA 落地的真实坑，教程已记录。

### Q19：小模型 SFT 生成退化（✅ 2026-08-06 已记录）

18.1M TinyStories 模型 Full-SFT（英文 alpaca，3 epoch）后：held-out assistant loss 6.41→3.62（-43.5%），但 greedy 生成退化为 `*`（token 13）重复循环。原因分析：18M 参数容量不足以学指令遵循格式（对比 Qwen3 LoRA 同数据规模 ppl 8.40→6.20 且生成质量明显提升），且 TinyStories 预训练分布与 alpaca 指令分布差异大。诚实记录：小模型 SFT 的 loss 下降不等于生成质量提升；Full-SFT 教学价值（优化器/scheduler/resume/assistant-only loss 全链路）已达成。

### Q20：Qwen3 chat template 空 think 块对训练/评测口径的影响（✅ 2026-08-06 已验证）

Qwen3 官方 chat template 对**最后一条 assistant 消息**无条件插入空 `<think>\n\n</think>\n\n` 块（训练数据如此渲染）。但 `add_generation_prompt=True` 的生成前缀默认**不包含**该块——若不显式 `enable_thinking=False`，训练与生成的前缀分布不一致，模型会在答案前产生噪声前缀（实测出现泰文乱码）。修复：生成时 `apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)`，与训练口径完全对齐。SFT 评测（held-out assistant loss）因使用同一模板+同一 mask 不受影响。
