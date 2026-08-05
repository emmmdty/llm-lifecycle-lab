# AGENTS.md

## 项目使命

本项目用于学习 LLM 从数据到部署的完整生命周期。目标是理解、实现、测试和验收，不追求最大模型、最大数据或业务演示。

## 开始前必读

每次任务必须先阅读：

1. `README.md`
2. `docs/00_PROJECT_AND_SERVER_AUDIT.md`
3. `docs/01_PROJECT_PLAN.md`
4. `docs/02_STAGE_TASKS_AND_ACCEPTANCE.md`
5. `docs/03_COMPLETE_TUTORIAL.md`
6. `docs/04_ENV_DATA_MODELS.md`
7. `docs/05_GPU_AND_LOCAL_SERVER_WORKFLOW.md`
8. `docs/06_OPEN_QUESTIONS_AND_SCALING.md`

然后明确当前只执行哪一个阶段。

跨阶段需求只记录，不顺手实现。与模型规模、数据规模匹配、训练效果评价相关的开放问题登记在 `docs/06_OPEN_QUESTIONS_AND_SCALING.md`，对应阶段执行时逐项验证并把结论回写。

## 每次任务先输出

- 当前阶段目标；
- 已有输入；
- 缺失事实；
- 修改范围；
- 测试方案；
- GPU 预算；
- 停止条件。

跨阶段需求只记录，不顺手实现。

## 当前阶段

当前执行正式开发前准备：项目初始化、服务器 fallback、环境与资产目录准备。

当前不编写训练代码，不启动 GPU 训练或推理，不修改服务器系统配置。模型、数据和环境准备只在用户明确授权时执行；未授权的后续训练、评测、量化和部署只记录，不顺手实现。完成条件是仓库初始化文件齐全、文档一致、服务器资产事实可追溯、下一阶段条件明确。

## 本地与服务器职责

### 本地

只做：

- 源码；
- 配置；
- 文档；
- synthetic fixture；
- 单元测试；
- Git。

本地不得保存真实模型、真实数据、checkpoint 或 GPU 运行日志。

### 服务器

主服务器：

```text
tongjiakai@gpu-5090
```

主服务器目录：

```text
/mnt/aidata/tongjiakai/llm-lifecycle-lab
```

5090 不可达时可使用 fallback 服务器：

```text
gpu-4090
/data/TJK/internship-projects/llm-lifecycle-lab
```

4090 上的可用 GPU 不固定。启动 GPU 任务前只做当前空闲显存和计算进程的最小检查，再选择可用 GPU。

服务器负责真实数据、模型、GPU smoke、训练、评测、量化和部署。阶段 0 已完成一次硬件、环境、磁盘、Git 和已有资产基线扫描，后续任务默认复用 `docs/00_PROJECT_AND_SERVER_AUDIT.md`，不得在每个新任务重复完整扫描。

## 服务器安全规则

- 非 sudo 用户；
- 不使用 sudo；
- 不修改驱动或系统 CUDA；
- 不终止其他用户进程；
- 不执行 GPU reset；
- 不修改他人的文件；
- 不扫描整个共享盘造成高 I/O；
- 不覆盖已有 Git 代理配置；
- 不把 secrets 写入仓库或日志。

## 服务器基线复用规则

- 不要在每个任务重复收集 GPU 型号、数量、driver、CUDA、磁盘总量、inode、全量 venv 版本或完整资产清单；
- 只有出现故障、硬件/环境变更、服务器路径变化、资源不足、或阶段验收明确需要新证据时，才重新定位对应事实；
- 启动 GPU 任务前只做与调度安全直接相关的最小检查，例如当前 GPU 空闲显存和当前计算进程；
- `gpu-5090` 使用 cpolar 动态隧道；SSH 前如连接失败，先运行 `cpolar-ssh-update` 更新端口。端口或 host key 变化不等于服务器硬件/环境变化，不触发完整基线重扫；
- `gpu-4090` 使用固定 cpolar 端口，不由 `cpolar-ssh-update` 管理；如果固定端口 TCP 拒绝连接，优先检查 4090 的 cpolar 服务/固定隧道转发，而不是重复改 SSH alias；
- `gpu-4090` 只作为 5090 不可达时的 fallback；不要把某个 GPU 编号写成长期可用资源；
- 新窗口应引用 `docs/00_PROJECT_AND_SERVER_AUDIT.md`，不要把完整审计输出重复粘贴进上下文。

## GPU 运行规则

正式训练前必须完成：

1. 单元测试；
2. 单 batch forward/backward；
3. 最多 5 step smoke；
4. 100-200 step 性能基准；
5. 基于实测速率估算时长。

限制：

- 优先单卡；
- 两卡只用于明确多卡学习；
- 默认不超过 8 小时；
- 硬上限 10 小时；
- 必须有 `max_steps` 或 `max_tokens`；
- 必须支持 checkpoint resume；
- Codex 不得自动启动正式训练。

## 环境规则

只使用 `uv`。服务器环境必须隔离：

```text
.venv-train
.venv-eval
.venv-quant
.venv-serve
```

禁止把 LLM Compressor、lm-eval vLLM extra 和 vLLM 强装到同一环境。修改依赖后必须运行并记录：

```bash
uv pip check
uv pip freeze
```

## 数据和模型规则

- 下载前检查数据集大小、文件、许可证和 split；
- 所有下载必须有样本、文件、token 或磁盘上限；
- 已有数据先盘点，不重复下载；
- test split 不进入训练；
- 同一 document/prompt 不跨 split；
- 数据必须有 manifest；
- 外部数据放 `data/`，外部模型放 `models/`；
- 自己的 checkpoint 放 `artifacts/` 或 `runs/`；
- 大文件不提交 Git。

## 代码规则

- Python 3.12；
- 使用类型标注；
- 配置和路径不可散落硬编码；
- 新功能必须有测试；
- 本地测试使用 synthetic fixture；
- 训练入口支持 dry-run、max steps、resume；
- 日志保存 command、config、environment、hardware、revision、seed 和 metrics；
- 不删除失败日志，除非它是不属于当前项目资产的残留临时文件且用户明确要求；
- 不通过 mock 结果声称真实训练成功。

## 验收方式

任务完成必须提供：

- 修改文件；
- 测试命令和输出；
- 服务器 smoke 命令，如当前阶段适用；
- 指标文件，如当前阶段适用；
- checkpoint 或产物位置，如当前阶段适用；
- 每条验收标准的证据；
- 失败与未完成项。

无证据不得标记完成。

## 教程规则

- 每个阶段验收通过后，必须在 `docs/tutorials/` 下撰写该阶段阶段性教程（中文编号命名，每阶段一篇）；
- 简单流程用 Mermaid，复杂框图用 TikZ 绘制并导出 PNG 插入（`.tex` 源文件与 PNG 一并提交）；
- 教程不包含个人网络因素（cpolar 等）、不暴露个人服务器信息、不写本地↔服务器同步说明；
- 环境内容必须写清 Python / uv / CUDA / torch 版本与隔离原因；
- 每篇教程必须记录该阶段遇到的真实问题与解决过程；
- 写作规范详见 `docs/tutorials/README.md`。

## Code Review Rules

- 不接受将 `data/`、`models/`、`artifacts/`、`runs/`、`logs/`、`.venv*` 或大型权重/数据文件提交到 Git。
- 不接受没有阶段边界的“一次性实现全项目”改动。
- 不接受没有测试证据的新增 Python 功能。
- 不接受把服务器真实训练、评测或下载结果写成未经验证的结论。
