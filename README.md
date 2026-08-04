# LLM Lifecycle Lab

LLM Lifecycle Lab 是一个面向学习和验收的 LLM 全生命周期实验仓库。项目用小数据、小模型和严格的阶段边界，练习从数据治理到部署评测的完整链路。

English summary: this repository is a staged learning lab for the LLM lifecycle, with local source development and remote GPU execution separated by design.

## 当前阶段

当前项目处于正式开发前准备阶段：仓库初始化和服务器基线已经开始建立，后续训练、评测、量化和部署仍按阶段推进。

当前停止条件：

- 本地仓库有清晰 README、许可证、GitHub 社区文件和忽略规则；
- 项目文档之间的路径、服务器、数据、模型和环境描述一致；
- 服务器已有资产完成只读盘点；
- 远端环境、数据和模型准备只在用户明确授权时执行；
- 不运行 GPU 训练、推理或 smoke test；
- 不修改服务器系统配置。

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
