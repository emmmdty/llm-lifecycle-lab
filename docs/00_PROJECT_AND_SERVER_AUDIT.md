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
