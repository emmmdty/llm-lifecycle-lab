# 环境、数据与模型说明

## 1. uv 环境

禁止把全部依赖安装到同一个环境。

当前审计事实：

- 本地使用 `uv` 和 Python 3.12；
- 5090 项目目录已有 `.venv-train`、`.venv-eval`、`.venv-quant`、`.venv-serve`；
- 5090 使用 `/home/tongjiakai/.local/bin/uv`，2026-07-30 已对四个环境运行 `uv pip check` 并保存 freeze；
- 4090 fallback 使用 `/home/TJK/.local/bin/uv`，项目目录为 `/data/TJK/internship-projects/llm-lifecycle-lab`；
- 4090 fallback 已创建 `.venv-train`：Python 3.12.13，ModelScope 1.39.0，`uv pip check` 通过，freeze 见服务器 `reports/4090-train-freeze.txt`；
- 服务器不应把真实环境依赖同步到本地。

### 训练环境

```bash
uv venv .venv-train --python 3.12
source .venv-train/bin/activate
```

用于：

- PyTorch；
- Transformers；
- Datasets；
- Accelerate；
- PEFT；
- TRL；
- bitsandbytes；
- tokenizer；
- TensorBoard。

### 评测环境

```bash
uv venv .venv-eval --python 3.12
```

2026-07-29 已审计：

```text
lm-eval 0.4.12
torch 2.13.0+cu130
CUDA runtime 13.0
transformers 5.14.1
Python 3.12.3
```

不要安装 `lm_eval[vllm]`。vLLM 通过 API 与评测环境连接。

### 量化环境

```bash
uv venv .venv-quant --python 3.12
```

主要安装 LLM Compressor，不和 vLLM 混装。

### 部署环境

```bash
uv venv .venv-serve --python 3.12
```

在干净环境安装 vLLM，让它解析匹配的 PyTorch/CUDA wheel。

### 环境验收

每个环境保存：

```bash
python --version
uv pip freeze
uv pip check
```

训练环境额外检查：

```python
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
```

---

## 2. 模型目录

```text
models/
├── Qwen3-0.6B-Base
├── Qwen3-0.6B
├── Qwen3.5-0.8B-Base
└── Qwen3.5-0.8B
```

第一次运行前先检查，不重复下载。

2026-07-30 5090 已有：

```text
Qwen3-0.6B-Base      1.2G
Qwen3-0.6B           1.5G
Qwen3.5-0.8B-Base    1.7G
Qwen3.5-0.8B         1.7G
```

这些目录已做基本文件存在检查；阶段 1 前仍需做 checksum、revision 和可加载性验证。

4090 fallback 已准备：

```text
models/Qwen3-0.6B          已移入项目目录，并保留原路径 symlink
models/Qwen3-0.6B-Base     已通过 ModelScope 下载
```

4090 上尚未做 Transformers 加载、checksum 全量复核或 GPU smoke。

基本完整性：

- `config.json`；
- tokenizer 或 processor 文件；
- `.safetensors`；
- 多 shard 模型的 index；
- 文件大小和 partial 文件。

模型下载示例：

```bash
modelscope download \
  --model Qwen/Qwen3-0.6B-Base \
  --local_dir models/Qwen3-0.6B-Base
```

外部模型只放 `models/`；自己训练的 checkpoint 放 `artifacts/` 或 `runs/`。

---

## 3. 数据集清单

### 从零预训练

```text
AI-ModelScope/TinyStories
modelscope/wikitext
```

### CPT

```text
AI-ModelScope/tigerbot-law-plugin
```

### SFT

```text
AI-ModelScope/alpaca-gpt4-data-zh
```

### Reward Model / DPO

```text
llamafactory/ultrafeedback_binarized
```

### GRPO

```text
AI-ModelScope/gsm8k
```

### 多模态

```text
lmms-lab/ChartQA
```

### 可选评测

```text
modelscope/ceval-exam
modelscope/hellaswag
```

### 服务器已有可选数据

```text
data/raw/wikimedia
```

该目录约 67GB，当前未验收，不是第一轮主线默认输入。后续若使用，必须补 manifest、许可证、split、schema、样本上限和 token 预算。

### 5090 primary 已准备 raw 数据

2026-07-30 已在 `/mnt/aidata/tongjiakai/llm-lifecycle-lab` 串行准备：

```text
data/raw/tinystories                 约 954M，README.md 和 data/*.parquet
data/raw/wikitext                    约 698M，wikitext-103-v1 zip 已下载并解压
data/raw/tigerbot-law-plugin          约 29M
data/raw/alpaca-gpt4-data-zh          约 31M
data/raw/ultrafeedback_binarized      约 194M
data/raw/gsm8k                        约 5.6M
data/raw/chartqa                      约 70M，当前 ModelScope snapshot 为 test parquet 和 metadata
data/raw/ceval-exam                   约 5.1M，已 materialize csv split
data/raw/hellaswag                    约 68M，已 materialize train/validation/test jsonl
```

Manifest、下载日志和环境记录：

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

这些是 raw 资产，不等于阶段 2 数据治理已完成。训练前仍需确认许可证、schema、split 防泄漏、抽样、去重、token 统计和处理后 manifest。`test` split 不得进入训练。

### 4090 fallback 已下载 raw 数据

```text
data/raw/tinystories
data/raw/wikitext
data/raw/tigerbot-law-plugin
data/raw/alpaca-gpt4-data-zh
data/raw/ultrafeedback_binarized
data/raw/gsm8k
data/raw/chartqa
```

Manifest：

```text
data/manifests/4090_download_manifest.json
```

这些只是 raw 资产准备结果，不等于阶段 2 数据治理已完成。训练前仍需完成许可证确认、schema 检查、有界抽样、split 防泄漏、token 统计和处理后 manifest。

注意：`modelscope/wikitext` 的 README 与 dataset script 对许可证描述不一致；阶段 2 必须先解决该记录冲突。

---

## 4. 下载原则

先查看数据集文件页、大小、许可证和 split，再下载。

示例：

```bash
modelscope download \
  --dataset AI-ModelScope/gsm8k \
  --local_dir data/raw/gsm8k
```

严禁：

- 无过滤下载大型网页语料；
- 看到数据集名称就默认适合训练；
- 下载前不检查大小；
- 把 test split 合并进训练；
- 同时并发下载多个大型数据集；
- 重复下载已有目录。

下载前至少执行：

```bash
df -hT /mnt/aidata
df -ih /mnt/aidata
du -sh models data 2>/dev/null
```

---

## 5. 数据集选择理由

### TinyStories

适合小模型快速学习连贯生成，便于验证预训练代码。

### Wikitext

约 522.66MB，规模可控，来自精选 Wikipedia 文章，适合正式教学预训练。

### tigerbot-law-plugin

约 29.87MB、55,895 行、Apache-2.0，适合构造小型中文领域 CPT。

### alpaca-gpt4-data-zh

中文 instruction 数据，适合学习 SFT 格式，不应用作纯文本 CPT。

### UltraFeedback Binarized

具有 prompt/chosen/rejected，适合 Reward Model 和 DPO。必须抽样并按 prompt 分组切分。

### GSM8K

约 5.90MB、MIT，答案可自动验证，比第一次直接使用 OpenR1-Math-220k 更合理。

### ChartQA

用图表问答学习 VLM 数据管线和微调，不需要大规模通用多模态数据。

---

## 6. 数据规模控制

Codex 在实现下载或处理逻辑时必须设置：

- 最大文件数；
- 最大样本数；
- 最大 token 数；
- 最大磁盘预算；
- dry-run；
- 已有文件跳过；
- partial 文件识别。

任何“下载全部”操作都必须先由人检查数据集总大小。
