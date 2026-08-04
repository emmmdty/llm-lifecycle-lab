# GPU、服务器和本地开发说明

## 1. 已知服务器信息

### Primary: 5090

```text
SSH:
tongjiakai@gpu-5090

连接方式:
cpolar 动态隧道。本地已有 `cpolar-ssh-update`，连接失败时先运行它更新端口。

项目目录:
/mnt/aidata/tongjiakai/llm-lifecycle-lab

磁盘:
/mnt/aidata
ext4，约 3.6T，总可用约 2.1T，inode 使用约 1%

GPU:
1 张 NVIDIA GeForce RTX 5090，32607 MiB 显存

Driver / CUDA:
driver 580.126.09，nvidia-smi CUDA 13.0

权限:
非 sudo 用户
```

### Fallback: 4090

当 5090 的 cpolar 隧道或 SSH 服务不可达时，可使用 4090 服务器继续环境、数据、模型和后续 GPU 教学任务：

```text
SSH:
gpu-4090

项目目录:
/data/TJK/internship-projects/llm-lifecycle-lab

资产根目录:
/data/TJK

GPU:
4 张 NVIDIA GeForce RTX 4090，单卡 24564 MiB 显存

Driver:
580.173.02
```

4090 上的可用 GPU 不固定。启动 GPU 任务前只做当前空闲显存和计算进程的最小检查，再选择可用 GPU。

2026-07-29 4090 fallback 已完成项目目录、`.venv-train`、基础模型和 raw 数据下载准备：

```text
/data/TJK/internship-projects/llm-lifecycle-lab/.venv-train
/data/TJK/internship-projects/llm-lifecycle-lab/models
/data/TJK/internship-projects/llm-lifecycle-lab/data/raw
/data/TJK/internship-projects/llm-lifecycle-lab/data/manifests/4090_download_manifest.json
```

这些准备不包含 GPU smoke、训练依赖完整安装、CUDA/BF16 验证或正式训练。

这些是 2026-07-29 的审计结果，作为后续阶段默认复用的硬件和服务器基线。不要在每个新任务重复完整扫描；只有出现故障、硬件/环境变更、资源不足或阶段验收明确要求新证据时，才重新检查相关子项。

## 2. 基线复用与最小检查

不能根据主机名猜测 GPU 情况；优先使用 `docs/00_PROJECT_AND_SERVER_AUDIT.md` 中已经记录的基线事实。

后续任务只在必要时做最小检查：

- SSH 失败时先运行 `cpolar-ssh-update`，不要因为端口或 host key 变化重扫服务器；
- 5090 不可达时可切换到 `gpu-4090`，但不要固定使用某个 GPU 编号；
- 启动 GPU 任务前检查当前空闲显存、GPU 利用率和计算进程；
- 修改环境前检查目标 venv 的 Python、Torch/CUDA runtime 和 `uv pip check`；
- 下载或处理数据前检查目标目录空间、inode、已有目录和 manifest；
- Git 同步前检查当前分支、远程和未跟踪资产冲突；
- 出现错误时只定位相关子系统，不重新扫描所有硬件和资产。

只记录与当前任务相关的检查结果，不擅自修改系统设置，不把完整基线重复粘贴到每个新窗口。

## 3. GPU 规则

- 启动 GPU 任务前执行最小 `nvidia-smi` 检查；
- 不杀死任何其他用户进程；
- 不执行 GPU reset；
- 不默认独占所有卡；
- 优先单卡；
- 两卡只用于明确的 DDP/FSDP 教学实验；
- 所有训练有 `max_steps` 或 `max_tokens`；
- smoke test 最多 5 step；
- 先跑 100–200 step 基准再估时；
- 默认 8 小时软上限；
- 10 小时硬上限；
- 异常退出保留日志；
- Codex 不自动开始正式训练。

## 4. 磁盘规则

只检查自己的目录，避免高 I/O 扫描整个共享盘。

```bash
df -hT /mnt/aidata
df -ih /mnt/aidata

du -sh \
  /mnt/aidata/tongjiakai/llm-lifecycle-lab/models \
  /mnt/aidata/tongjiakai/llm-lifecycle-lab/data \
  /mnt/aidata/tongjiakai/llm-lifecycle-lab/artifacts \
  /mnt/aidata/tongjiakai/llm-lifecycle-lab/runs \
  2>/dev/null
```

不得下载 TB 级数据。原始数据和 checkpoint 必须有保留策略。

## 5. 本地与服务器职责

### 本地

- Codex/IDE 开发；
- Python、Shell、YAML 和 Markdown；
- 单元测试；
- synthetic fixtures；
- Git commit 和 code review。

本地不保存：

- 真实模型；
- 真实数据；
- checkpoint；
- GPU 环境。

### 服务器

- ModelScope 下载；
- 数据预处理；
- GPU smoke；
- 正式训练；
- 评测；
- 量化；
- vLLM/llama.cpp；
- 运行日志和 checkpoint。

## 6. Git 同步

推荐流程：

本地：

```bash
uv run pytest
git status
git add .
git commit -m "implement stage X"
git push
```

服务器：

```bash
ssh tongjiakai@gpu-5090
cd /mnt/aidata/tongjiakai/llm-lifecycle-lab

git status
git fetch --all --prune
git pull --ff-only
```

服务器使用 GitHub 代理镜像。Codex 应先检查：

```bash
git remote -v
git config --global --get-regexp '^url\..*\.insteadof$' || true
```

不得覆盖既有全局代理配置。

5090 不可达且使用 4090 fallback 时：

```bash
ssh gpu-4090
cd /data/TJK/internship-projects/llm-lifecycle-lab

git status
git fetch --all --prune
git pull --ff-only
```

## 7. SCP 同步

Git 暂时不可用时，可从本地只上传源码和文档：

```bash
scp -r \
  src configs tests docs AGENTS.md pyproject.toml \
  tongjiakai@gpu-5090:/mnt/aidata/tongjiakai/llm-lifecycle-lab/
```

上传前确认命令中不包含：

```text
models/
data/
artifacts/
runs/
logs/
.venv*/
.env
```

不要使用会删除服务器端数据目录的同步参数。

## 8. 服务器上的 Git 修改

若必须在服务器调试并修改代码：

1. 创建独立 branch；
2. 只提交源码、配置和小型报告；
3. 不提交数据、模型、环境或 checkpoint；
4. 推送后在本地审查；
5. 不使用 `git reset --hard` 随意清除未审查修改。

## 9. 事故处理

失败后依次记录：

1. 完整命令；
2. 第一个异常；
3. traceback；
4. GPU 状态；
5. 环境 freeze；
6. 数据样本/schema；
7. 最小复现；
8. 修复；
9. 同规模回归测试。

不要直接增加 GPU、batch 或训练时长掩盖根因。
