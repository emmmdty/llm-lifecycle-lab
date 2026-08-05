"""Stage 6 CPT: comparison-table tests (torch-free)."""

from __future__ import annotations

from cpt.compare import DEFAULT_EVAL_PROMPTS, build_comparison, summarize_comparison


def sample_result(loss: float, blocks: int = 100) -> dict:
    import math

    return {"val_loss": loss, "val_ppl": math.exp(loss), "val_blocks": blocks}


def test_build_comparison_domain_and_general() -> None:
    base = {
        "domain_val": sample_result(2.0),
        "general_wikitext": sample_result(1.5),
        "general_tinystories": sample_result(1.2),
    }
    cpt = {
        "domain_val": sample_result(1.8),
        "general_wikitext": sample_result(1.6),
        "general_tinystories": sample_result(1.3),
    }
    table = build_comparison(base, cpt)
    assert table["domain_val"]["kind"] == "domain"
    assert table["domain_val"]["delta_ppl"] < 0
    assert table["domain_val"]["base_ppl"] == base["domain_val"]["val_ppl"]
    assert table["general_wikitext"]["kind"] == "general"
    assert table["general_wikitext"]["delta_ppl"] > 0
    assert table["general_wikitext"]["delta_pct"] > 0


def test_summarize_comparison_acceptance_flags() -> None:
    base = {
        "domain_val": sample_result(2.0),
        "general_wikitext": sample_result(1.5),
        "general_tinystories": sample_result(1.2),
    }
    cpt = {
        "domain_val": sample_result(1.8),
        "general_wikitext": sample_result(1.6),
        "general_tinystories": sample_result(1.3),
    }
    summary = summarize_comparison(build_comparison(base, cpt))
    assert summary["domain_ppl_improved"] is True
    assert summary["domain_ppl_delta"] < 0
    assert summary["general_ppl_max_degradation"] > 0
    assert summary["general_ppl_mean_delta"] > 0
    assert summary["acceptance"]["domain_improved"] is True
    assert summary["acceptance"]["general_degradation_quantified"] is True


def test_summarize_comparison_no_domain() -> None:
    base = {"general_wikitext": sample_result(1.5)}
    cpt = {"general_wikitext": sample_result(1.5)}
    summary = summarize_comparison(build_comparison(base, cpt))
    assert summary["domain_ppl_improved"] is False
    assert summary["domain_ppl_delta"] is None
    assert summary["acceptance"]["domain_improved"] is False


def test_default_eval_prompts_include_domain_and_general() -> None:
    assert len(DEFAULT_EVAL_PROMPTS) == 3
    assert any("合同" in p for p in DEFAULT_EVAL_PROMPTS)
    assert any(p.isascii() for p in DEFAULT_EVAL_PROMPTS)
