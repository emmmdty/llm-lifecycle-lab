# LLM Lifecycle Lab

LLM Lifecycle Lab 是一个面向学习和验收的 LLM 全生命周期实验仓库。项目用小数据、小模型和严格的阶段边界，练习从数据治理到部署评测的完整链路。

English summary: this repository is a staged learning lab for the LLM lifecycle, with local source development and remote GPU execution separated by design.

## 当前阶段

本项目定位为 **LLM 全链路教学项目**（与 MiniMind、动手学 LLM 的定位差异与自我约束见 [docs/01 §1.1](docs/01_PROJECT_PLAN.md)）：主线教学模型规模按"尽可能大"原则结合实测数据决策——minimind_dataset 实测 ~1.40B tokens，用户确认**严格 D≈20N → ~70M**（决策过程见 [docs/01 §3.1](docs/01_PROJECT_PLAN.md) 与 [docs/06 Q21](docs/06_OPEN_QUESTIONS_AND_SCALING.md)），贯穿 SFT、RM、DPO、GRPO、统一评测、量化、部署全链路，Qwen3-0.6B 作为强基座对照。

项目已完成数据治理（阶段 2）、tokenizer（阶段 3）、TinyStories 快速预训练（阶段 4，18.1M 模型、全语料 392M tokens、实测 10.8 分钟完成闭环）、Wikitext 正式教学预训练（阶段 5，Q1 缩放实验 + Q2 决策 + 80M token 正式训练）、Qwen3 CPT（阶段 6，rank-2 LoRA 573K 可训练参数、tigerbot-law 3.91M domain token、domain ppl -31.6%、通用无退化、7.4 分钟完成）和 SFT（阶段 7，tiny Full-SFT + Qwen3 LoRA/QLoRA-SFT 双实验、assistant-only loss、prompt 分组数据、merge 一致性、Full vs LoRA vs QLoRA 资源对比）。当前执行**阶段 8（主线从零预训练）**：minimind_dataset 数据治理、~70M 规模决策（严格 D≈20N，2026-08-10 用户确认）、训练、与小模型资产（保留为教程对比项）的对照评测。阶段编号已重排：RM→10、DPO→11、GRPO→12、多模态→13、评测→14、量化→15、部署→16、最终验收→17（见 [docs/00](docs/00_PROJECT_AND_SERVER_AUDIT.md) 2026-08-10 补充）。

阶段 7 停止条件已满足：

- 本地单元测试通过（107 passed + 1 skipped，torch-free 核心逻辑全测，服务器 117 passed）；
- 服务器按顺序完成 dry-run、5 step smoke、150 step 基准（含 resume 连续性）、人工确认后正式训练；
- 数据：中文 alpaca-gpt4-zh 与英文 alpaca-cleaned（cc-by-4.0）均按 prompt 分组切分，test 排除，token+assistant-mask 流 + manifest；
- held-out assistant-only loss 下降：tiny -43.5%（6.41→3.62）、LoRA -14.3%（2.13→1.83）、QLoRA -11.5%（2.13→1.88）；
- 固定 50 prompt 前后对比（reports/sft-compare-*.json）；LoRA merge 前后一致（BF16 基座），QLoRA 需去量化后 merge（PEFT 已知警告）；
- Full vs LoRA vs QLoRA 峰值显存 3.59 / 14.13 / 7.43GB，tokens/s 668K / 29.1K / 18.5K；
- 正式训练合计约 11 分钟，远低于 8h 上限；command/config/environment/hardware/revision/seed/metrics 全部记录在 `runs/` 与 `reports/`；
- 诚实记录：tiny 18.1M SFT 后生成退化（容量限制，Q19）；QLoRA NF4 基座 merge 有损的正确流程（Q18）。

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
