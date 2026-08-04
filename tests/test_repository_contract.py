from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_required_root_files_exist() -> None:
    required_files = [
        "README.md",
        "LICENSE",
        ".gitignore",
        ".gitattributes",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "SECURITY.md",
        "AGENTS.md",
        "pyproject.toml",
    ]

    missing = [path for path in required_files if not (ROOT / path).is_file()]

    assert missing == []


def test_required_github_templates_exist() -> None:
    required_files = [
        ".github/PULL_REQUEST_TEMPLATE.md",
        ".github/ISSUE_TEMPLATE/config.yml",
        ".github/ISSUE_TEMPLATE/stage-task.yml",
        ".github/ISSUE_TEMPLATE/audit-or-docs.yml",
    ]

    missing = [path for path in required_files if not (ROOT / path).is_file()]

    assert missing == []


def test_agents_references_existing_project_docs() -> None:
    agents = read_text("AGENTS.md")
    required_references = [
        "README.md",
        "docs/00_PROJECT_AND_SERVER_AUDIT.md",
        "docs/01_PROJECT_PLAN.md",
        "docs/02_STAGE_TASKS_AND_ACCEPTANCE.md",
        "docs/03_COMPLETE_TUTORIAL.md",
        "docs/04_ENV_DATA_MODELS.md",
        "docs/05_GPU_AND_LOCAL_SERVER_WORKFLOW.md",
    ]

    missing_references = [path for path in required_references if path not in agents]
    missing_files = [path for path in required_references if not (ROOT / path).is_file()]

    assert missing_references == []
    assert missing_files == []


def test_readme_links_existing_project_docs() -> None:
    readme = read_text("README.md")
    required_references = [
        "docs/00_PROJECT_AND_SERVER_AUDIT.md",
        "docs/01_PROJECT_PLAN.md",
        "docs/02_STAGE_TASKS_AND_ACCEPTANCE.md",
        "docs/03_COMPLETE_TUTORIAL.md",
        "docs/04_ENV_DATA_MODELS.md",
        "docs/05_GPU_AND_LOCAL_SERVER_WORKFLOW.md",
        "LICENSE",
    ]

    missing_references = [path for path in required_references if path not in readme]

    assert missing_references == []


def test_mit_license_uses_expected_holder() -> None:
    license_text = read_text("LICENSE")

    assert "MIT License" in license_text
    assert "Copyright (c) 2026 emmmdty" in license_text


def test_gitignore_excludes_large_and_local_assets() -> None:
    gitignore = read_text(".gitignore")
    required_patterns = [
        ".venv/",
        ".venv-*",
        ".env",
        "data/",
        "models/",
        "artifacts/",
        "runs/",
        "logs/",
        "requirements-*.lock.txt",
        "*.safetensors",
        "*.gguf",
        "*.parquet",
    ]

    missing_patterns = [pattern for pattern in required_patterns if pattern not in gitignore]

    assert missing_patterns == []


def test_docs_do_not_record_removed_fineweb_asset() -> None:
    tracked_docs = [
        "README.md",
        "AGENTS.md",
        "docs/00_PROJECT_AND_SERVER_AUDIT.md",
        "docs/01_PROJECT_PLAN.md",
        "docs/02_STAGE_TASKS_AND_ACCEPTANCE.md",
        "docs/03_COMPLETE_TUTORIAL.md",
        "docs/04_ENV_DATA_MODELS.md",
        "docs/05_GPU_AND_LOCAL_SERVER_WORKFLOW.md",
    ]

    offenders = []
    for path in tracked_docs:
        if not (ROOT / path).is_file():
            continue
        if "fineweb" in read_text(path).lower():
            offenders.append(path)

    assert offenders == []
