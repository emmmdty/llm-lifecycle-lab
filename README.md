# LLM Lifecycle Lab

LLM Lifecycle Lab 是一个面向学习和验收的 LLM 全生命周期实验仓库。项目用小数据、小模型和严格的阶段边界，练习从数据治理到部署评测的完整链路。

English summary: this repository is a staged learning lab for the LLM lifecycle, with local source development and remote GPU execution separated by design.

## 当前阶段

项目已完成数据治理（阶段 2）、tokenizer（阶段 3）、TinyStories 快速预训练（阶段 4，18.1M 模型、全语料 392M tokens、实测 10.8 分钟完成闭环）、Wikitext 正式教学预训练（阶段 5，Q1 缩放实验 + Q2 决策 + 80M token 正式训练）和 Qwen3 CPT（阶段 6，rank-2 LoRA 573K 可训练参数、tigerbot-law 3.91M domain token、domain ppl -31.6%、通用无退化、7.4 分钟完成）。当前执行阶段 7（SFT）的开发准备。

阶段 6 停止条件已满足：

- 本地单元测试通过（85 passed + 1 skipped，torch-free 核心逻辑全测，服务器 96 passed）；
- 服务器按顺序完成 dry-run、单 batch、5 step smoke、150 step 基准（30K tokens/s、MFU 49.8%）、resume 连续性验证与人工确认后正式训练；
- domain perplexity 7.015 → 4.795（-31.6%），通用退化被量化（identical-framing 下无退化，框架诊断见 reports/cpt-framing-diagnostic.json）；
- adapter 重载往返一致（重载 ppl vs 训练末次 val 差异 ~1e-3，BF16 最后一位非确定性）；
- 正式训练 446.5 秒（7.4 分钟），远低于 3 小时预算；
- command/config/environment/hardware/revision/seed/metrics 全部记录在 `runs/` 与 `reports/`。

## 本地与服务器职责

本地 WSL2 仓库只保存源码、配置、文档、synthetic fixture、单元测试和 Git 元数据。本地不保存真实数据、真实模型、checkpoint、运行日志或 GPU 环境。

真实数据、模型、环境、评测、量化和部署默认在 5090 服务器执行：

```text
SSH: tongjiakai@gpu-5090
目录: /mnt/aidata/tongjiakai/llm-lifecycle-lab
```

当 5090 因 cpolar 隧道或 SSH 服务不可达时，可使用 4090 服务器作为 fallback：

```text
SSH: gpu-4090
目录: /data/TJK/internship-projects/llm-lifecycle-lab
```

4090 上的可用 GPU 不固定。启动任何 GPU 任务前，只做当前空闲显存和计算进程的最小检查，再选择可用 GPU。

已审计的服务器事实见 [docs/00_PROJECT_AND_SERVER_AUDIT.md](docs/00_PROJECT_AND_SERVER_AUDIT.md)。

## 文档索引

- [docs/00_PROJECT_AND_SERVER_AUDIT.md](docs/00_PROJECT_AND_SERVER_AUDIT.md)：阶段 0 本地与服务器审计事实。
- [docs/01_PROJECT_PLAN.md](docs/01_PROJECT_PLAN.md)：项目目标、阶段路线、模型与数据路线。
- [docs/02_STAGE_TASKS_AND_ACCEPTANCE.md](docs/02_STAGE_TASKS_AND_ACCEPTANCE.md)：每个阶段的任务和验收标准。
- [docs/03_COMPLETE_TUTORIAL.md](docs/03_COMPLETE_TUTORIAL.md)：LLM 生命周期教程说明。
- [docs/04_ENV_DATA_MODELS.md](docs/04_ENV_DATA_MODELS.md)：环境、数据和模型规范。
- [docs/05_GPU_AND_LOCAL_SERVER_WORKFLOW.md](docs/05_GPU_AND_LOCAL_SERVER_WORKFLOW.md)：本地与 GPU 服务器工作流。
- [docs/06_OPEN_QUESTIONS_AND_SCALING.md](docs/06_OPEN_QUESTIONS_AND_SCALING.md)：模型规模、数据规模与效果评价的开放问题登记册（跨阶段验证计划）。
- [AGENTS.md](AGENTS.md)：Codex 和其他 coding agent 的仓库级执行规则。

## 开发约束

- Python 使用 3.12。
- Python 工具链使用 `uv`。
- 一次只做一个阶段。
- 新功能必须有测试。
- 本地测试只使用 synthetic fixture。
- 服务器正式训练前必须先完成单元测试、单 batch、最多 5 step smoke 和 100-200 step 性能基准。
- Codex 不自动启动正式训练。

## 本地验证

```bash
uv lock
uv run pytest -q
git diff --check
git status --short --ignored
```

## 许可证和第三方资产

本仓库源码和文档使用 [MIT License](LICENSE)。

第三方数据、模型、checkpoint、日志和量化产物不属于本仓库授权范围，也不得提交到 Git。服务器上的外部模型和数据必须按各自许可证、revision、来源和 manifest 管理。
