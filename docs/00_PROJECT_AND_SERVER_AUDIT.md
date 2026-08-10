# 阶段 0 项目和服务器审计

审计日期：2026-07-29

## 当前阶段目标

完成正式开发前准备：本地仓库初始化、文档一致性、GitHub 社区文件、许可证、忽略规则、服务器硬件/环境/资产只读盘点。

本阶段不下载数据或模型，不启动 GPU 训练、推理或 smoke test，不修改服务器系统配置。

## 本地仓库事实

```text
路径: /home/tjk/myProjects/internship-projects/llm-lifecycle-lab
分支: main
状态: 尚无初始 commit
远程: https://github.com/emmmdty/llm-lifecycle-lab.git
```

本地只保存源码、配置、文档、synthetic fixture、单元测试和 Git 元数据。真实数据、模型、checkpoint、日志和环境目录必须由 `.gitignore` 排除。

## 服务器事实

### 5090 primary baseline

```text
SSH: tongjiakai@gpu-5090
hostname: server-System-Product-Name
user: tongjiakai
项目目录: /mnt/aidata/tongjiakai/llm-lifecycle-lab
```

`gpu-5090` 通过 cpolar 动态隧道连接。后续任务如 SSH 失败，先在本地运行 `cpolar-ssh-update` 更新 `~/.ssh/config` 中的端口；端口或 host key 变化只属于隧道状态变化，不代表服务器硬件或环境基线变化。

GPU：

```text
数量: 1
型号: NVIDIA GeForce RTX 5090
显存: 32607 MiB
driver: 580.126.09
nvidia-smi CUDA: 13.0
compute capability: (12, 0)
```

这是后续阶段默认复用的服务器硬件基线。不要在每个新任务重复完整扫描；只有出现故障、硬件/环境变更、资源不足或阶段验收明确要求新证据时，才重新检查相关子项。

磁盘：

```text
/mnt/aidata: ext4, 3.6T total, 2.1T available, 41% used
inode: 233M total, 1% used
```

### 4090 fallback baseline

2026-07-29，`gpu-5090` 的 cpolar 公网端口可达但 SSH banner 无响应，当前可使用 `gpu-4090` 作为 fallback 服务器继续项目准备。

2026-07-30 连接注记：`gpu-4090` 使用固定 cpolar 端口，不由 `cpolar-ssh-update` 管理。本地 `ssh -G gpu-4090` 指向 `1.tcp.vip.cpolar.top:12644`，DNS 可解析到 `116.211.150.54`，但 TCP 连接返回 `Connection refused`；同一 cpolar 域名上的 a6000 端口和 5090 当前端口可连通。因此该次失败发生在 4090 固定 cpolar 监听/转发层，不是本地 SSH alias、用户名、密钥或 known_hosts 问题。

```text
SSH: gpu-4090
hostname: 4090-1
user: TJK
项目目录: /data/TJK/internship-projects/llm-lifecycle-lab
资产根目录: /data/TJK
```

GPU：

```text
数量: 4
型号: NVIDIA GeForce RTX 4090
显存: 24564 MiB each
driver: 580.173.02
```

磁盘：

```text
/data: ext4, 58T total, 36T available at audit time
```

4090 上的可用 GPU 不固定。启动 GPU 任务前只做当前空闲显存和计算进程的最小检查，再选择可用 GPU；不要在计划或文档中写死 GPU 编号。

调度工具：

```text
Slurm/PBS 常见命令未在 PATH 中发现。
```

## 5090 服务器已有资产

模型目录：

```text
models/Qwen3-0.6B-Base      1.2G
models/Qwen3-0.6B           1.5G
models/Qwen3.5-0.8B-Base    1.7G
models/Qwen3.5-0.8B         1.7G
```

这些目录包含 `config.json`、tokenizer/processor 文件、`.safetensors` 权重和 Apache-2.0 license 文件。阶段 1 之前仍需用脚本做 checksum、文件大小和可加载性复核。

数据目录：

```text
data/raw/tinystories                 954M，README.md 和 data/*.parquet
data/raw/wikitext                    698M，wikitext-103-v1 zip 已下载并解压
data/raw/tigerbot-law-plugin          29M
data/raw/alpaca-gpt4-data-zh          31M
data/raw/ultrafeedback_binarized     194M
data/raw/gsm8k                       5.6M
data/raw/chartqa                      70M，当前 ModelScope snapshot 为 test parquet 和 metadata
data/raw/ceval-exam                  5.1M，已 materialize csv split
data/raw/hellaswag                    68M，已 materialize train/validation/test jsonl
data/raw/wikimedia                    67G，既有可选/未验收资产
```

`wikimedia` 数据保留在服务器上，当前标记为未验收/可选资产。它不是当前文本主线的默认输入；后续若使用，必须先补 manifest、许可证、split、schema、样本上限和 token 预算。

5090 准备记录：

```text
data/manifests/5090_raw_assets_manifest.json
logs/downloads/prepare_5090_assets_scoped_20260730-093758.log
logs/downloads/materialize_5090_scripted_raw_20260730-094651.log
reports/5090_asset_summary.txt
reports/5090-train-freeze.txt
reports/5090-eval-freeze.txt
reports/5090-quant-freeze.txt
reports/5090-serve-freeze.txt
```

`artifacts/` 和 `runs/` 目录已补齐为空目录。上述 raw 数据仍不是阶段 2 后的 governed 数据；训练前必须完成许可证、schema、split、抽样、去重、token 预算和处理后 manifest。

## 4090 fallback 已准备资产

2026-07-29，用户明确授权在 `gpu-4090` 上准备环境、模型和数据，未启动 GPU 训练或推理。

项目源码已同步到：

```text
/data/TJK/internship-projects/llm-lifecycle-lab
```

模型：

```text
models/Qwen3-0.6B          1.5G，已从 /data/TJK/DEE/SARGE/models/Qwen/Qwen3-0.6B 移入项目目录
models/Qwen3-0.6B-Base     1.2G，已通过 ModelScope 下载
```

原路径 `/data/TJK/DEE/SARGE/models/Qwen/Qwen3-0.6B` 已保留为指向项目目录的兼容 symlink。

数据：

```text
data/raw/tinystories                 约 1.0G，仅 README.md 和 data/*.parquet
data/raw/wikitext                    约 698M，wikitext-103-v1 zip 已校验并解压 train/validation/test
data/raw/tigerbot-law-plugin          约 29M
data/raw/alpaca-gpt4-data-zh          约 31M
data/raw/ultrafeedback_binarized      约 203M
data/raw/gsm8k                        约 5.9M
data/raw/chartqa                      约 73M，当前 ModelScope snapshot 为 test parquet 和 metadata
```

Manifest 与下载日志位于服务器项目目录：

```text
data/manifests/4090_download_manifest.json
logs/downloads/
reports/4090_asset_summary.txt
reports/4090-train-freeze.txt
```

注意：`modelscope/wikitext` 的 README 和 dataset script 中许可证描述不一致，训练前必须在数据治理阶段明确最终许可证记录。

## 环境事实

### 5090 primary

服务器项目目录下已有隔离环境：

```text
.venv-train
.venv-eval
.venv-quant
.venv-serve
```

已导入核对：

```text
.venv-train: Python 3.12.3, torch 2.13.0+cu130, CUDA runtime 13.0, ModelScope 1.39.0, hf_transfer 0.1.9
.venv-eval:  torch 2.13.0+cu130, CUDA runtime 13.0, lm-eval 0.4.12
.venv-quant: torch 2.12.0+cu130, CUDA runtime 13.0, llmcompressor 0.12.0
.venv-serve: torch 2.11.0+cu130, CUDA runtime 13.0, vLLM 0.26.0
```

可复现 `uv` 调用方式：

```text
/home/tongjiakai/.local/bin/uv
```

2026-07-30 已对 `.venv-train`、`.venv-eval`、`.venv-quant`、`.venv-serve` 运行 `uv pip check`，均通过；freeze 记录见 `reports/5090-*-freeze.txt`。

### 4090 fallback

4090 使用 `/home/TJK/.local/bin/uv`，缓存目录：

```text
UV_CACHE_DIR=/data/TJK/uv-cache
PIP_CACHE_DIR=/data/TJK/pip-cache
```

已创建：

```text
/data/TJK/internship-projects/llm-lifecycle-lab/.venv-train
```

环境核对：

```text
uv 0.11.8
Python 3.12.13
ModelScope 1.39.0
uv pip check: passed
uv pip freeze: reports/4090-train-freeze.txt
```

当前只安装下载和数据准备所需轻量包；尚未安装 PyTorch、Transformers、PEFT、TRL、vLLM 或量化工具。

## 风险和未确认项

- 本地和服务器仓库都尚无初始 commit，服务器目录包含未跟踪的大资产，不能直接 `git add .`。
- `wikimedia` 数据只有原始下载事实，尚无项目 manifest，不能用于训练验收。
- 5090 raw 数据已准备 manifest，但尚未完成阶段 2 数据治理，不能直接作为训练验收数据。
- 模型目录看起来完整，但尚未做 checksum、revision、加载和 processor/tokenizer 一致性验证。
- 4090 fallback 的 `.venv-train` 目前只完成下载工具环境；训练依赖和 CUDA/BF16 验证仍属于后续阶段。
- 4090 fallback 的可用 GPU 编号不固定；启动 GPU 任务前必须按当前空闲显存和计算进程动态选择。
- 4090 fallback 已下载的 raw 数据不能直接进入训练；阶段 2 仍需完成许可证、schema、split、抽样、去重、token 预算和 manifest 加强验收。

## 阶段 1 可执行条件

- 本地初始化文件和文档通过测试；
- 大资产被 `.gitignore` 排除；
- 本地初始 commit 推送到 GitHub；
- 服务器在不覆盖未跟踪资产的前提下同步源码；
- 明确阶段 1 只做环境审计/修复和 CPU-only 资产完整性检查，不启动正式训练；
- 阶段 1 复用本文件中的硬件/磁盘/Git/资产基线，只补充和环境修复直接相关的最小事实。

## 2026-08-04 补充：初始 commit 与源码同步

本地仓库完成初始 commit 并推送到 GitHub，服务器在不覆盖未跟踪资产的前提下同步源码，阶段 1 可执行条件已满足：

- 本地：`tests/test_repository_contract.py` 通过（7 passed），大资产由 `.gitignore` 排除，初始 commit 已推送至 `https://github.com/emmmdty/llm-lifecycle-lab.git`（分支 `main`）。
- 服务器：通过 `git fetch` + checkout 同步源码，`models/`、`data/`、`logs/`、`reports/`、`artifacts/`、`runs/`、`.venv-*` 和 `requirements-*-lock.txt` 等未跟踪资产保持原状，未被覆盖或删除。
- 服务器 GitHub 访问：`tongjiakai` 用户已配置与本地一致的 gh CLI 认证（`~/.local/bin/gh`、`~/.config/gh/hosts.yml` 0600）和 git 身份；实测直连 github.com 可用，无代理镜像，git fetch/push 验证通过。
- 2026-08-05 补充：github.com 直连在服务器上间歇性不可达，已配置长期 fallback——`origin` fetch URL 指向 `gh-proxy.com` 镜像（读走镜像），push URL 保持直连 GitHub（gh 凭据仅对 github.com 生效）；镜像完整性以 commit 哈希校验。

## 2026-08-05 补充：阶段 4 TinyStories 快速预训练完成

阶段 4 在 5090 上完成：18,108,928 参数（18.1M）Decoder-only causal LM 在 TinyStories train 全语料（392,186,497 tokens，含 BOS/EOS）上训练 11,969 步，GPU 用时 646.8 秒（预算 2 小时）。

服务器新增资产：

```text
data/processed/tinystories/tokens/tinystories-bpe-16k/
  train.bin 1.57G（392,186,497 tokens int32 流，1,799,248 docs）
  validation.bin 12M（3,131,630 tokens，15,389 docs）
  train.json / validation.json（meta：revision、license、special_ids、环境）

runs/20260805-151300/
  run.json（command/config/environment/hardware/revision/seed/summary）
  metrics.jsonl（11,969 行：loss/lr/grad_norm/tokens_s/step_time/val/peak_mem）
  samples.jsonl（init 随机乱码 vs final 连贯故事）
  checkpoints/（step-1000 … step-11969 共 12 个 + latest.pt）

runs/20260805-145445/  smoke + resume 连续性证据（resume_continuity）
logs/train/formal-20260805.log、prepare-20260805.log
reports/20260805-151300.json
```

关键指标：平均 606K tokens/s、0.054s/step、峰值显存 1.22GB；train loss 9.807→1.525；val loss 3.192→1.665（ppl 5.28）；resume 后 loss 逐位一致。

环境补充：`.venv-train` 于本阶段追加安装 `pytest>=8.0`（运行测试用），`uv pip check` 通过；freeze 未变（pytest 非训练依赖）。

阶段 5 可执行条件：本阶段训练框架（prepare/train/generate、checkpoint/resume、记录）已就绪；阶段 5 仅需新增 30M–60M 模型配置与 Wikitext 数据准备。

## 2026-08-05 补充：阶段 5 Wikitext 正式教学预训练完成

阶段 5 在 5090 上完成：Q1 多规模实验（5.14M / 18.1M / 65.4M 各 1 epoch）、Q2 规模决策（正式 5.32M 匹配 D≈20N + 26.8M 欠训练对照）、Wikitext 80M token 正式预训练、登记册 Q3/Q8/Q10/Q12 验证。

### 新增服务器资产

```text
data/processed/wikitext/tokens/tinystories-bpe-32k/
  train.bin 354M（92,311,043 tokens int32 流，647,821 docs；BOS/EOS 计入）
  validation.bin 1.2M（297,414 tokens，2,071 docs）
  train.json / validation.json（meta：revision=wikitext-103-v1 snapshot 2026-07-29、
  license=CC BY-NC 4.0、tokenizer=tinystories-bpe-32k、git_commit）

runs/20260805-172251/    Q1 5.14M TinyStories 1 epoch（11969 步，val 1.971）
runs/20260805-173744/    Q1 65.4M TinyStories 1 epoch（11969 步，val 1.439）
runs/20260805-181744/    Q3 5.14M TinyStories 3 epoch（35907 步，val 1.769）
runs/20260805-201228/    Wiki 5.32M 正式（2441 步/80M token，val 4.7394，ppl 114.37）
runs/20260805-201504/    Wiki 26.76M 欠训练对照（2441 步/80M token，val 3.7685，ppl 43.32）
runs/20260805-201132/    Wiki 5.32M 基准（150 步）+ resume bit-check（step-75）
logs/train/（q1-*-1epoch.log、q3-5m-3epoch.log、formal-wiki-*.log、bench-*）
reports/stage5-analysis.json
```

阶段 4 既有数据点复用：18.1M TinyStories 1 epoch → val 1.665（ppl 5.28）。

### 阶段 5 关键结果

- **Q1**：`L(N)=13.26·N^(-0.124)`，R²=0.998，α 落在文献区间（Chinchilla ~0.1）；D≈20N 框架适用（详见 docs/06 第 2 节）。
- **Q2 决策**：正式 5.32M（t/p=15.0，val ppl 114.37）+ 26.8M 对照（t/p=3.0，val ppl 43.32）。26.8M 绝对 loss 更低源于 5× 算力投入；按固定算力最优分配其最优规模约 10M，t/p=3.0 处于数据受限区（详见 docs/06）。
- **Q3**：5M 1→3 epoch val 1.971→1.769；18M×1 epoch（1.665）仍优于 5M×3 epochs（1.769）→ 固定数据下加参数 > 加 epoch。
- **Q10 MFU**（5090 FP16 dense 380 TFLOPS 基准）：5M=3.4%、18M≈17%、64M=16.6%、Wiki-5.3M=4.1%、Wiki-26.8M=9.3%。小模型受 kernel 启动开销限制，MFU 低属预期。
- **Q12 embedding 占比**：TS-5M=42%、TS-18M=48%、TS-64M=13%、Wiki-5.3M=81%、Wiki-26.8M=65%——32k 词表 × 小 hidden 时 embedding 严重挤占参数。
- **Q8**：全部运行 val-train gap 为正且小（0.017–0.147，无过拟合）；生成多样性（distinct 4-gram ratio）：TinyStories 模型 1.0，Wikitext 模型 0.59–0.63（重复退化，与人工查看样本一致）。
- 数据事实：32k BPE 将 Wikitext train 编码为 92.31M token（manifest 16k 估算 80M），训练预算仍按 80M（max_steps 2441）。

### 阶段 5 验收证据

- **无 NaN/Inf**：全部运行 dry-run loss finite、metrics 无异常值（每个 run.json 的 dry_run_loss/loss_finite）。
- **validation loss 下降**：5.32M：10.40→4.74（ppl 114.4，24 个验证点）；26.76M：9.18→3.77（ppl 43.3）；5M/64M/3epoch 同理（metrics.jsonl）。
- **resume 可用**：Wiki-5.32M 基准从 step-75 checkpoint 恢复（同 resolved config），step 76–91 与原始运行**逐位一致**，后续步骤出现 BF16 最后一位非确定性（~1e-5 相对误差，与阶段 4 现象相同；阶段 4 记录的"逐位一致"应理解为约 5–6 位小数一致）。resume 机制（模型/优化器/采样器/RNG 状态恢复）与损失连续性已验证。
- **GPU 预算**：全部单卡；单次最长 45.9 分钟（Q3 3-epoch），远低于 8h 默认上限；无需放宽。
- **显存解释**：参数 BF16 2B/param + 梯度 2B/param + AdamW（fp32 主副本 + 2 个 fp32 动量）12B/param ≈ 16B/param，再加 activation。实测峰值：5.32M=1.5GB、26.76M=2.0GB、64M=3.12GB（16B×N 分别约 85MB/0.43GB/1.05GB，剩余为 activation）。
- **运行记录**：每次运行 run.json 含 command/config/environment/hardware/revision/seed/summary（loss、tokens/s、step time、峰值显存、avg_mfu），metrics.jsonl 逐 step 记录（含 mfu、val_train_gap）。
- **Q2/Q1 结论**：docs/06 第 2 节与登记册已回写；docs/01 模型路线已按决策更新。
- **失败与未完成**：Q7/Q13 未验证（对照混淆/规模不可比，已记录）；Q8 任务级评测留待阶段 12；Q3 只做了 5M 单模型。

阶段 6 可执行条件：预训练框架与规模决策方法已就绪；Qwen3 CPT 直接进入阶段 6 任务。

## 2026-08-05 补充：阶段 6 Qwen3 CPT 完成

阶段 6 在 5090 上完成：tigerbot-law 中文领域 LoRA-CPT（Qwen3-0.6B-Base + rank-2 LoRA），domain/general held-out 对比评测，adapter 重载验证，Q8/Q9 顺带验证。

### 新增服务器资产

```text
data/processed/tigerbot-law-cpt/tokens/qwen3/
  domain_train.bin + .json（3,912,160 tokens，52,810 docs）
  domain_val.bin + .json（208,584 tokens，2,648 docs）
data/processed/general-wikitext/tokens/qwen3/validation.bin + .json（261,118 tokens）
data/processed/general-tinystories/tokens/qwen3/validation.bin + .json（3,076,291 tokens）
data/manifests/tigerbot-law-cpt.json（切分策略/seed/token 统计/许可证）

runs/smoke6/20260805-210911/    5 step smoke
runs/bench6/20260805-211005/    150 step 基准 + resume 连续性验证（step-75 恢复）
runs/20260805-211848/           正式训练 seed 42（360 步，7.4 分钟）
runs/20260805-212917/           Q9 双 seed 对照 seed 43（360 步）
reports/cpt-compare-formal.json  Base vs CPT 对比（ppl + 生成）
reports/cpt-framing-diagnostic.json  EOS 前缀框架诊断矩阵
logs/cpt/*.log
```

### 阶段 6 关键结果

- **数据**：tigerbot-law 治理产物（train+validation，test 为空且排除）按 title 文档分组 95/5 重切（seed 42，同一 document 不跨 split），Qwen3 tokenizer 编码：domain_train 3.91M token、domain_val 209K；general held-out 复用治理 validation parquet 重新编码（wikitext 261K、tinystories 3.08M），未下载任何新数据。
- **6ND LoRA 规模决策**：D=3.91M → unique 读法 N=D/20≈196K；seen 读法（3 epoch）N=3D/20≈587K。rank-2（q/k/v/o，573,440 可训练参数 = base 0.096%）匹配 seen 读法（0.98×），相对 unique 读法放宽 2.9×（base 已预训练、adapter 只学分布偏移，已记录理由）。
- **正式训练**（seed 42，run 20260805-211848）：360 步 / 11.8M token / 446.5 秒（7.4 分钟，预算 3h）；26,421 tokens/s、1.24s/step、MFU 49.8%、峰值显存 14.1GB；train loss 1.907→1.496。
- **Base vs CPT（同一脚本/held-out/tokenizer，100 块 seed 1234）**：domain_val ppl 7.015→4.795（**-31.6%**）；wikitext 17.176→14.495（-15.6%）；tinystories 7.013→6.137（-12.5%）。
- **通用退化结论（诚实解读）**：identical-framing 下 general 未见退化，反而略降；框架诊断（EOS-as-BOS 前缀）显示 base 在该框架下被惩罚 0.7–1.2 ppl，CPT 通过训练适应了该框架；用 no_bos 框架保守对照，CPT 仍不劣于 base（domain 6.75→5.83，wikitext 15.88→15.34，tinystories 6.16→6.12）。结论：rank-2 LoRA-CPT 带来显著领域改善且无可量化通用退化。
- **adapter 重载**：PeftModel.from_pretrained 重载后 domain ppl 4.795 vs 训练末次 val 4.801（~1e-3 差异，BF16 最后一位非确定性），往返一致。
- **Q8**：固定 3 prompt（法律/中文日常/英文经济）Base vs CPT 生成对照入 reports/cpt-compare-formal.json；多样性（distinct 4-gram）0.632→0.655。
- **Q9**：seed 42 vs 43 全配置对照，domain ppl 4.801 vs 4.799（<0.1%）、general <1%，seed 敏感性低。

### 环境变更（2026-08-05，阶段 6）

- `.venv-train` 追加 `ziglang==0.16.0`：服务器无系统 C 编译器（gcc/clang 均无）且系统 Python 无开发头文件（/usr/include/python3.12/Python.h 缺失），torch 2.13 原生 op bmm_outer_product（Qwen3 rotary 内部使用）走 triton 实现，triton 首次运行需编译 cuda_utils 驱动模块。
- 解决：`uv pip install ziglang` + `.venv-train/bin/zig-cc` 包装器（`zig cc`，把 `-l:libcuda.so.1` 译为绝对路径输入）；`uv python install 3.12` 获取带头文件的 CPython，`.triton-cc/sitecustomize.py` 重定向 sysconfig include（该目录已在服务器 `.git/info/exclude` 忽略，不进 Git）；`libcuda.so.1` 复制进 triton nvidia lib 目录。triton cuda_utils 一次性编译成功并缓存。
- `uv pip check` 通过；freeze 已更新（reports/5090-train-freeze.txt）。

## 2026-08-06 补充：阶段 7 SFT 完成

阶段 7 在 5090 上完成：SFT 数据准备（中文 + 英文 prompt 分组）、三实验训练（tiny Full-SFT / Qwen3 LoRA / Qwen3 QLoRA）、assistant-only loss、packing、resume、merge 一致性、Full vs LoRA vs QLoRA 资源对比。

### 新增服务器资产

```text
data/processed/alpaca-sft-zh/     中文 SFT（source: alpaca-gpt4-zh）
  tokens/qwen3/train.bin + train.mask.bin（15,389 docs / 2,432,864 token）
  tokens/qwen3/validation.bin + validation.mask.bin（811 docs / 129,624 token）
  prompts-50.json（固定对比 prompt，seed 2026，来自 val）
data/processed/alpaca-sft-en/     英文 SFT（source: alpaca-cleaned，cc-by-4.0，44MB 新下载）
  tokens/tinystories-bpe-16k/train.bin + .mask.bin（12,825 docs / 2,825,578 token）
  tokens/tinystories-bpe-16k/validation.bin + .mask.bin（675 docs / 155,875 token）
  prompts-50.json
data/manifests/alpaca-sft-zh.json、alpaca-sft-en.json、alpaca-cleaned.json
data/processed/alpaca-cleaned/{train,validation,test}.parquet（12K/1.5K/1.5K 治理产物）

runs/smoke7/                     三实验 5-step smoke
runs/bench7/                     三实验 150-step bench + step-75 resume 连续性验证
runs/20260806-082602/            tiny Full-SFT 正式（129 步，12.6s，val ppl 37.48）
runs/20260806-082619/            Qwen3 LoRA-SFT 正式（222 步，250s，val ppl 6.20）
runs/20260806-083036/            Qwen3 QLoRA-SFT 正式（222 步，393s，val ppl 6.58）
reports/stage7-eval-summary.json 阶段 7 评测汇总
reports/sft-eval-*.json          held-out assistant-only loss（before/after）
reports/sft-compare-{tiny,lora,qlora}.json  固定 50 prompt 前后对比
reports/sft-merge-*.json         LoRA/QLoRA merge 一致性
logs/sft/*.log                   bench/formal/compare/resume 日志
```

### 阶段 7 关键结果

- **数据**：中文 alpaca-gpt4-zh 治理产物按 user prompt 分组重切（train 15,389 / val 811，治理 test 因与 train 有 2 处 prompt 重叠而排除，不进入训练）；英文 alpaca-cleaned 新下载（cc-by-4.0，44MB）治理后按 prompt 分组重切（12,825/675）。两条流均为 token 流 + 并行 int8 assistant-mask 流（1=assistant token）。同一 prompt 不跨 split（分组切分 + 单测验证）。固定 50 prompt 从各自 val 采样（seed 2026，选择规则记录在 manifest）。
- **小模型语言匹配决策（Q16）**：小模型（英文预训练）用英文 alpaca-cleaned；Qwen3 用中文 alpaca-gpt4-zh（Qwen3 中英双语）。中文在小模型 16k BPE 下 token 化碎片化严重（8 字→30 token vs 7 英文词→7 token）。
- **LoRA rank 决策**：SFT 学指令遵循（新行为分布）非分布偏移，按领域先例 rank 8（q/k/v/o，2.29M 可训练参数 = base 0.38%），QLoRA 同 rank 对比。
- **assistant-only loss**：mask 流实现，prompt 段 label=-100（单测验证：prompt token 不产生 loss）。packing 沿用 packed stream + BlockSampler。
- **三实验（同一脚本、同一 val blocks、同一有效 batch token 口径 32K/step）**：

| 实验 | 训练 token | tokens/s | step/s | 峰值显存 | MFU | held-out assistant loss（before→after） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny Full-SFT 18.1M | 8.45M | 668K | 0.10 | 3.59GB | 38.2% | 6.41→3.62（-43.5%，ppl 610→37.5） |
| Qwen3 LoRA r8 | 7.27M | 29.1K | 1.13 | 14.13GB | 54.8% | 2.13→1.83（-14.3%，ppl 8.40→6.20） |
| Qwen3 QLoRA r8 (NF4) | 7.27M | 18.5K | 1.77 | 7.43GB | 22.0% | 2.13→1.88（-11.5%，ppl 8.40→6.58） |

- **resume 连续性**：三实验均从 step-75 checkpoint 恢复，loss 逐位差异 ≤1.1e-3（BF16 最后一位非确定性，与阶段 4/5/6 一致）。QLoRA checkpoint 只存 adapter 状态（NF4 基座权重不入 checkpoint，resume 时从磁盘重建）。
- **merge 一致性（Q17/Q18）**：BF16 LoRA merge 前后 held-out loss 差 3.3e-4、生成 5/5 一致；QLoRA merge 进 NF4 基座有损（ppl 6.65→8.88，PEFT 已知警告），**去量化到 BF16 基座后 merge 一致**（loss 差 2.1e-4、生成 10/10）。QLoRA 落地真实坑，已记录。
- **chat template（Q20）**：Qwen3 模板对最后一条 assistant 消息插空 `<think>\n\n</think>\n\n`；生成时需 `enable_thinking=False` 对齐训练口径，否则产生噪声前缀（实测泰文乱码）。小模型无 chat template，自建 `user: X [EOS] assistant: Y` 文本模板（16k/32k BPE 无角色 token，无法加特殊 token 以免改词表破坏 checkpoint）。
- **诚实记录（Q19）**：tiny 18.1M Full-SFT 后 held-out loss 大幅下降但生成退化为 `*` 重复（18M 容量不足以学指令遵循；对比 Qwen3 同数据规模生成质量明显提升）。小模型 SFT 的 loss 下降不等于生成质量提升。
- **GPU 预算**：全部单卡；正式训练合计 12.6s + 250s + 393s ≈ 11 分钟，远低于 8h 上限。
- **框架**：TRL 1.9.2 SFTTrainer 可导入但未使用（与项目自写 loop 框架约定不一致，且需自定义 mask/checkpoint/MFU）；自写 src/sft/ 沿用阶段 4/5/6 模式（dry-run/max-steps/resume/run.json/metrics.jsonl/checkpoint v1 格式）。
- **失败与未完成**：tiny 生成退化未解决（容量限制，非训练 bug）；QLoRA NF4 基座 merge 有损（已给出正确流程）；TRL SFTTrainer 未做完整集成验证（仅 import + 决策记录）。

阶段 8 可执行条件：SFT 管线（messages→template→mask 流→训练→eval→merge）已就绪；RM 阶段直接消费 ultrafeedback_binarized（prompt 分组治理产物已在阶段 2 完成）。

## 2026-08-06 补充：阶段 0-7 代码审查与修复

2026-08-06 对阶段 0-7 全部代码做了一次 subagent code review，修复以下问题（git 记录见对应 commit）：

| # | 严重度 | 问题 | 修复 |
| --- | --- | --- | --- |
| 1 | 严重 | `BlockSampler`/`validation_offsets` 上界差一（off-by-one）：offset 可取到 `stream_len - seq_len`，label 切片 `stream[o+1:o+seq_len+1]` 越界 1 元素，numpy memmap 静默截断后 `np.stack` 随机崩溃 | 最大 offset 改为 `stream_len - seq_len - 1`；构造守卫改为 `stream_len <= seq_len` 时报错（label shift 需 seq_len+1 个 token）；测试同步更新 |
| 2 | 中等 | LoRA/QLoRA 的 MFU 用 12ND 虚高 2-3 倍（冻结 base 无 dW，实际约 4ND/token） | `cpt/train.py`、`sft/train.py` 改为 4ND（PEFT）/ 6ND（Full-SFT）；`cpt/lora.py` 的 FLOP 注释与 `estimate_cpt_flops` 同步修正；历史报告的 MFU 数字为旧口径，以本文档为准 |
| 3 | 中等 | `pretrain/run.py` main 先 `_setup_logging(None)` 再 `cmd_train` 二次 `basicConfig` 静默 no-op，训练日志文件为空 | 与 cpt/sft 一致：`command != "train"` 时才预置 logging |
| 4 | 中等 | merge-check 的合并前生成在 train 模式运行（dropout 激活），`generation_exact_match` 不可信 | `generate_conversation`/`generate_text` 生成前 `model.eval()`、生成后 `model.train()` |
| 5 | 建议 | govern official 策略 budget+token_cap 同时设置时 cap 覆盖 budget | 先 budget 裁剪再对裁剪结果 cap |
| 6 | 建议 | resume 与 CLI override 冲突时报错不友好 | 错误信息补充"resume requires the exact same config; CLI overrides are not supported" |
| 7 | 建议 | `ckpt_every=0` 时连最终 checkpoint 都不存（违反"必须支持 resume"） | 三处训练器最终 checkpoint 无条件保存 |
| 8 | 建议 | Embedding 初始化覆盖 `padding_idx` 零值 | 初始化后对 `padding_idx` 行显式清零 |
| 9 | 建议 | cpt `ModelConfig` 未校验 Qwen3 上下文上限、source_corpus 硬编码、sft tokenizer revision 恒为 None、死代码 `merge_adapter` | 补校验；`source_corpus` 移入配置（默认值保持兼容）；revision 回退到模型路径说明；删除死代码 |
| 10 | 轻微 | pretrain `generate()` 的窗口化分支是死代码、group_by 截断无对账计数、govern reader 按 suffix 而非 spec.reader 分派 | 已记录，未改（行为无影响；generation 窗口化留待阶段 12 专门处理） |

审查结论：数据泄漏防护（cpt 按 document / sft 按 prompt 分组切分）、grad_accum 归一化、checkpoint resume 的 sampler/RNG 恢复、SFT assistant-mask 移位语义、QLoRA adapter-only checkpoint 恢复、cpt/sft eval 与 train validate 的一致性——均未发现问题。

**MFU 口径说明（重要）**：阶段 4/5 小模型（全参训练）MFU 用 6ND，口径正确不变。阶段 6/7 的 LoRA/QLoRA 报告 MFU 原按 12ND 计算（虚高约 2-3 倍），本次修正为 4ND 口径；`runs/` 内历史 run.json 的 `mfu` 字段未重算，解读时乘以约 1/3（LoRA 4ND/12ND）才是 4ND 口径数值。docs/06 Q10 的小模型 MFU 数据（5M=3.4% 等）为全参训练，不受影响。

## 2026-08-10 补充：项目定位升级与阶段重排

用户决策：本项目是 LLM 全链路教学项目，主线模型应为 5090 单卡可承载的**尽可能大**的从零预训练模型（100M–200M 级，64M 只是示例数字），原小模型资产保留为教程对比项。原规划（RM/DPO/GRPO 均以 Qwen3 为唯一载体）升级为：**主线模型贯穿全链路，Qwen3 作为强基座对照**。

### 阶段编号重排（历史记录中的旧编号不追溯修改）

```text
原：阶段 8=RM, 9=DPO, 10=GRPO, 11=多模态, 12=评测, 13=量化, 14=部署, 15=验收
新：阶段 8=主线预训练（100M–200M）, 9=主线 SFT, 10=RM, 11=DPO, 12=GRPO,
    13=多模态, 14=统一评测, 15=量化, 16=部署, 17=最终验收
```

本文档历史补充中出现的"阶段 8（RM）""阶段 12（统一评测）"等引用均为当时的旧编号，对应新编号 10 / 14。

### 规模上限事实（修正 docs/06 §3 旧错误）

docs/06 §3 原记"8h 上限约 875M 参数 / 17.5B token"与同文 §4.3"1B/20B ≈ 21 天"矛盾（875M 按实测吞吐需约 380h，错约 10 倍）。本次修正为：按实测 ~66 TFLOPS 与 D≈20N 外推，**8h 上限约 128M / 2.6B token；24h 硬上限约 200M / 4B token**（完整计算与 docs/06 §4.3 一致）。64M / 5 亿 token ≈ 50 分钟 与 18M 全语料 10.8 分钟 的既有外推点保持正确。

### 主线预训练约束与决策（docs/01 §3.1、docs/06 Q21 登记）

- 时间：单次默认 ≤8h，硬上限 <24h（物理上限：8h → ~128M / 2.6B token；24h → ~200M / 4B token）；
- 硬盘：raw ~9.5GB + int32 token 流 ≤6GB + checkpoint 保留 ≤16 个，合计 ≤20GB；
- 显存：非约束（32GB 下 200M 级峰值约 6GB）；
- 语料：minimind_dataset pretrain（ModelScope 标注 CC-BY-NC-4.0 / HF 标注 Apache-2.0，冲突以更严格者记录），raw 已下载，须按阶段 2 流程治理；
- **规模决策（2026-08-10 用户确认）**：实测 ~1.40B tokens（Qwen3 口径）→ 严格 D≈20N → **N≈70M**（治理后按选定 tokenizer 实测修正，预期 60M–85M）。

### 与经典项目的定位关系（docs/01 §1.1 详细）

MiniMind（产品/复现项目）与动手学 LLM（工具使用教程）与本项目（亲手实现 + 实测证据 + 全链路 + 验收制教学）定位不同，非重复造轮子；差异在"过程证据"（数据治理、缩放律验证、统一评测、量化/部署基准、失败分析）。自我约束：不抄代码、教程每章带实测数据与失败案例、不把"再训一个同规模模型"当作验收目标。

## 2026-08-10 补充：阶段 8 主线预训练 raw 数据准备完成

minimind_dataset（gongjy/minimind_dataset，ModelScope）预训练 raw 已下载至 `data/raw/minimind_dataset`（仅预训练相关文件，未下载 SFT/DPO/RLAIF 等无关文件）：

```text
pretrain_t2t.jsonl       8,275,074,893 字节（8.27GB），8,468,827 行，sha256 31efc9a6...cc9d ✓
pretrain_t2t_mini.jsonl  1,241,043,656 字节（1.24GB），1,270,238 行，sha256 6dd6716c...5560c ✓
README.md / .gitattributes
```

关键事实：

- schema 统一 `{"text": str}`，zh + en（中文为主）；minimind 原始用法是直接作为 text→next-token 文本流；
- **许可证冲突**：ModelScope yaml 标注 CC-BY-NC-4.0，HF（jingyaogong/minimind_dataset）标注 Apache-2.0——以更严格者记录，manifest 已注明（data/manifests/minimind-dataset.json）；
- **规模决策关键事实与结果**：Qwen3 tokenizer 抽样 5000 条（chars/token=1.65），主文件外推 **~1.40B tokens**——低于 200M 模型 D≈20N 所需 4B token；严格 D≈20N 匹配为 ~70M，128M–200M 将处于数据受限区（t/p≈7–11，minimind 先例 t/p≈7.8）。**2026-08-10 用户确认：严格 D≈20N，N≈70M**（数据受限选项未采用，理由见 docs/01 §3.1 与 docs/06 Q21）；
- 样本含大量指令/对话风格文本，治理时需检查质量分布；mini 与主文件是否重叠需治理时核对；
- 下载日志：logs/downloads/prepare_minimind_dataset_20260810-143016.log；manifest：data/manifests/minimind-dataset.json；
- 硬盘预算更新：主线预训练资产（raw + token 流 + checkpoint）合计 ≤20GB。

数据治理（去重、抽样、held-out 切分、tokenizer 选定、token 流）属阶段 8 正式执行任务，等待 tokenizer 决策后进行（规模已决策：~70M，严格 D≈20N）。
