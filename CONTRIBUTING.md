# Contributing

本项目按阶段开发。任何贡献都必须说明当前阶段、修改范围、验证命令和剩余风险。

## 基本流程

1. 阅读 `README.md`、`AGENTS.md` 和 `docs/` 中对应阶段文档。
2. 明确本次只修改哪一个阶段。
3. 本地只提交源码、配置、文档、synthetic fixture 和测试。
4. 真实数据、模型、checkpoint、日志和环境目录不得提交。
5. 修改后运行：

```bash
uv lock
uv run pytest -q
git diff --check
git status --short --ignored
```

## 服务器规则

服务器用于真实数据、模型、GPU smoke、训练、评测、量化和部署。启动任何长时间或 GPU 任务前，必须先给出命令、工作目录、预期输出和停止条件。

