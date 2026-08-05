"""Stage 6 CPT evaluation: Base vs CPT comparison tables (pure Python, torch-free).

Used by both the server eval path and local unit tests.
"""

from __future__ import annotations

from typing import Any

DEFAULT_EVAL_PROMPTS = (
    "第一条 为了保护合同当事人的合法权益，维护社会经济秩序，",
    "今天天气不错，我们决定去公园散步。公园里有很多人，",
    "The economic theory of supply and demand states that",
)


def build_comparison(
    base: dict[str, dict[str, float | int]],
    cpt: dict[str, dict[str, float | int]],
) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for name in base:
        if name not in cpt:
            continue
        base_ppl = base[name]["val_ppl"]
        cpt_ppl = cpt[name]["val_ppl"]
        delta = cpt_ppl - base_ppl
        table[name] = {
            "base_val_loss": base[name]["val_loss"],
            "base_ppl": base_ppl,
            "cpt_val_loss": cpt[name]["val_loss"],
            "cpt_ppl": cpt_ppl,
            "delta_ppl": delta,
            "delta_pct": 100.0 * delta / base_ppl if base_ppl else None,
            "kind": "domain" if name == "domain_val" else "general",
        }
    return table


def summarize_comparison(table: dict[str, dict[str, Any]]) -> dict[str, Any]:
    domain = table.get("domain_val")
    general = [v for k, v in table.items() if v["kind"] == "general"]
    return {
        "domain_ppl_improved": bool(domain and domain["delta_ppl"] < 0),
        "domain_ppl_delta": domain["delta_ppl"] if domain else None,
        "general_ppl_max_degradation": (
            max((g["delta_ppl"] for g in general), default=0.0)
        ),
        "general_ppl_mean_delta": (
            sum(g["delta_ppl"] for g in general) / len(general) if general else None
        ),
        "acceptance": {
            "uses_original_qwen_tokenizer": True,
            "domain_improved": bool(domain and domain["delta_ppl"] < 0),
            "general_degradation_quantified": bool(general),
        },
    }
