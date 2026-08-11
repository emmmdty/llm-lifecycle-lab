"""Stage 2 data governance: dataset registry."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    license: str
    revision: str
    upstream: str
    reader: str
    transform: str
    split_strategy: str
    pattern: str = ""
    official_files: dict[str, tuple[str, ...]] = field(default_factory=dict)
    projection: tuple[str, ...] | None = None
    group_key: str | None = None
    partitions: dict[str, int] | None = None
    budget: int | None = None
    token_cap: int | None = None
    seed: int = 42
    fracs: tuple[float, float, float] = (0.8, 0.1, 0.1)
    notes: tuple[str, ...] = ()


DATASETS: dict[str, DatasetSpec] = {
    "tinystories": DatasetSpec(
        name="tinystories",
        license="cdla-sharing-1.0",
        revision="modelscope snapshot 2026-07-29",
        upstream="AI-ModelScope/TinyStories",
        reader="parquet",
        transform="tinystories",
        split_strategy="official",
        official_files={
            "train": ("tinystories/data/train-*.parquet",),
            "validation": ("tinystories/data/validation-*.parquet",),
        },
        seed=42,
        notes=("官方无 test split；训练子集保留官方切分。",),
    ),
    "wikitext": DatasetSpec(
        name="wikitext",
        license="CC BY-NC 4.0",
        revision="wikitext-103-v1 (modelscope snapshot 2026-07-29)",
        upstream="modelscope/wikitext",
        reader="text",
        transform="wikitext",
        split_strategy="official",
        official_files={
            "train": ("wikitext/wikitext-103/wiki.train.tokens",),
            "validation": ("wikitext/wikitext-103/wiki.valid.tokens",),
            "test": ("wikitext/wikitext-103/wiki.test.tokens",),
        },
        token_cap=80_000_000,
        seed=42,
        notes=(
            "README 记录 CC BY-NC 4.0，modelscope dataset script 描述不一致，以 README 为准。",
            "train 按 token 上限 80M 有界抽样，匹配阶段 5 训练预算。",
        ),
    ),
    "minimind-pretrain": DatasetSpec(
        name="minimind-pretrain",
        license="CC-BY-NC-4.0（ModelScope yaml 标注；HF 标注 Apache-2.0，以更严格者记录）",
        revision="gongjy/minimind_dataset ModelScope snapshot 2026-08-10",
        upstream="gongjy/minimind_dataset",
        reader="jsonl",
        transform="minimind",
        split_strategy="shuffle",
        pattern="minimind_dataset/pretrain_t2t.jsonl",
        fracs=(0.98, 0.02, 0.0),
        seed=42,
        notes=(
            "阶段 8 主线预训练语料：minimind_dataset pretrain_t2t.jsonl（8.27GB / 8,468,827 行，sha256 已校验，manifest 见 data/manifests/minimind-dataset.json）。",
            "全量治理（无 budget/cap）：8.27GB 本身是有界输入；shuffle 切分 98/2，test 比例为 0（预训练无 test 概念，validation 作 held-out）。",
            "按行/document 为最小单元切分（minimind 数据无标题/文档分组字段，行即最小独立单元）。",
            "minimind 原始用法是直接作为 text→next-token 文本流；本项目治理为 shuffle 切分 + manifest，并记录差异。",
            "mini 文件（pretrain_t2t_mini.jsonl）与主文件重叠核对在治理前置脚本完成，重叠则不纳入。",
        ),
    ),
    "tigerbot-law": DatasetSpec(
        name="tigerbot-law",
        license="Apache-2.0",
        revision="modelscope snapshot 2026-07-29",
        upstream="AI-ModelScope/tigerbot-law-plugin",
        reader="jsonl",
        transform="tigerbot",
        split_strategy="shuffle",
        pattern="tigerbot-law-plugin/tigerbot-laws-plugin.jsonl",
        token_cap=8_000_000,
        seed=42,
        fracs=(0.95, 0.05, 0.0),
        notes=("CPT 语料，正文 token 上限 8M，取 5% 为领域验证集。",),
    ),
    "alpaca-gpt4-zh": DatasetSpec(
        name="alpaca-gpt4-zh",
        license="CC BY NC 4.0",
        revision="modelscope snapshot 2026-07-29",
        upstream="AI-ModelScope/alpaca-gpt4-data-zh",
        reader="csv",
        transform="alpaca",
        split_strategy="shuffle",
        pattern="alpaca-gpt4-data-zh/train.csv",
        budget=18_000,
        seed=42,
        fracs=(0.8, 0.1, 0.1),
        notes=("非商用许可，仅用于本项目学习。总预算 18K：train 14.4K / val 1.8K / test 1.8K。",),
    ),
    "alpaca-cleaned": DatasetSpec(
        name="alpaca-cleaned",
        license="cc-by-4.0",
        revision="AI-ModelScope/alpaca-cleaned snapshot 2026-08-06",
        upstream="AI-ModelScope/alpaca-cleaned",
        reader="json_array",
        transform="alpaca",
        split_strategy="shuffle",
        pattern="alpaca-cleaned/alpaca_data_cleaned.json",
        budget=15_000,
        seed=42,
        fracs=(0.8, 0.1, 0.1),
        notes=(
            "英文指令数据（yahma/alpaca-cleaned 镜像），用于小模型 Full-SFT 的语言匹配数据。",
            "阶段 2 治理为 shuffle 切分；阶段 7 SFT 准备时按 prompt 分组重切（train+val 参与）。",
        ),
    ),
    "gsm8k": DatasetSpec(
        name="gsm8k",
        license="MIT",
        revision="modelscope snapshot 2026-07-29",
        upstream="AI-ModelScope/gsm8k",
        reader="parquet",
        transform="gsm8k",
        split_strategy="official",
        official_files={
            "train": ("gsm8k/main/train-*.parquet",),
            "test": ("gsm8k/main/test-*.parquet",),
        },
        budget=1_750,
        seed=42,
        notes=(
            "test split 仅作评测，不进训练。",
            "答案含 #### 最终数字，供 GRPO 解析。",
        ),
    ),
    "ultrafeedback-binarized": DatasetSpec(
        name="ultrafeedback-binarized",
        license="Apache-2.0",
        revision="llamafactory/ultrafeedback_binarized 2026-07-29",
        upstream="llamafactory/ultrafeedback_binarized",
        reader="json_array",
        transform="ultrafeedback",
        split_strategy="group_by",
        pattern="ultrafeedback_binarized/train.json",
        group_key="instruction",
        partitions={
            "rm_train": 4_000,
            "rm_val": 1_000,
            "dpo_train": 7_000,
            "dpo_val": 1_000,
            "eval": 1_000,
        },
        seed=42,
        notes=(
            "按 instruction 分组哈希切分，同一 prompt 不跨 partition。",
            "官方 test.json 保留为独立评测资产，不进训练。",
        ),
    ),
    "chartqa": DatasetSpec(
        name="chartqa",
        license="Apache-2.0",
        revision="swift/ChartQA (ms-swift mirror of lmms-lab/ChartQA) 2026-08-05",
        upstream="swift/ChartQA",
        reader="parquet",
        transform="chartqa",
        split_strategy="official",
        official_files={
            "train": ("chartqa-swift/data/train-*.parquet",),
            "validation": ("chartqa-swift/data/val-*.parquet",),
            "test": ("chartqa-swift/data/test-*.parquet",),
        },
        projection=("query", "label", "human_or_machine"),
        budget=5_000,
        seed=42,
        notes=(
            "swift 镜像 schema 为 query/label/human_or_machine，已适配；label 为 list。",
            "train 按阶段 11 预算上限 5K 抽样（计划 2K-5K）。",
            "图像保留在 raw，processed 只存文本字段；阶段 11 从 raw 加载 image。",
            "原 lmms-lab/ChartQA 快照仅 test 且无许可标注，已由 swift/ChartQA 完整镜像替换。",
        ),
    ),
    "hellaswag": DatasetSpec(
        name="hellaswag",
        license="Apache-2.0",
        revision="HF Rowan/hellaswag 2026-07-29",
        upstream="Rowan/hellaswag",
        reader="jsonl",
        transform="hellaswag",
        split_strategy="official",
        official_files={
            "train": ("hellaswag/hellaswag_train.jsonl",),
            "validation": ("hellaswag/hellaswag_val.jsonl",),
            "test": ("hellaswag/hellaswag_test.jsonl",),
        },
        seed=42,
        notes=("仅作统一评测用，train 不进训练。",),
    ),
    "ceval-exam": DatasetSpec(
        name="ceval-exam",
        license="cc-by-nc-sa-4.0",
        revision="ceval-exam snapshot 2026-07-29",
        upstream="AI-ModelScope/ceval-exam",
        reader="csv",
        transform="ceval",
        split_strategy="official",
        official_files={
            "dev": ("ceval-exam/dev/*.csv",),
            "val": ("ceval-exam/val/*.csv",),
            "test": ("ceval-exam/test/*.csv",),
        },
        seed=42,
        notes=("仅作统一评测用，不进训练。",),
    ),
}
