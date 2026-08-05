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
