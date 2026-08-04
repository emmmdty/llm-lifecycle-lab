"""Stage 2 data governance: record transforms."""

from __future__ import annotations

from typing import Callable

Transform = Callable[[dict], dict | None]


def alpaca(row: dict) -> dict | None:
    instruction = str(row.get("instruction") or "").strip()
    extra_input = str(row.get("input") or "").strip()
    output = str(row.get("output") or "").strip()
    if not instruction or not output:
        return None
    user = instruction if not extra_input else f"{instruction}\n\n{extra_input}"
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": output},
        ]
    }


def tigerbot(row: dict) -> dict | None:
    content = str(row.get("content") or "").strip()
    if not content:
        return None
    return {"text": content, "title": str(row.get("title") or "").strip()}


def gsm8k(row: dict) -> dict | None:
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip()
    if not question or "####" not in answer:
        return None
    return {"question": question, "answer": answer}


def tinystories(row: dict) -> dict | None:
    text = str(row.get("text") or "").strip()
    if not text:
        return None
    return {"text": text}


def wikitext(row: dict) -> dict | None:
    text = str(row.get("text") or "").strip()
    if len(text) < 10:
        return None
    return {"text": text}


def ultrafeedback(row: dict) -> dict | None:
    instruction = str(row.get("instruction") or "").strip()
    chosen = str(row.get("chosen") or "").strip()
    rejected = str(row.get("rejected") or "").strip()
    if not instruction or not chosen or not rejected:
        return None
    return {"instruction": instruction, "chosen": chosen, "rejected": rejected}


def chartqa(row: dict) -> dict | None:
    question = str(row.get("question") or "").strip()
    if not question:
        return None
    return {
        "question": question,
        "answer": str(row.get("answer") or "").strip(),
        "type": str(row.get("type") or "").strip(),
    }


def ceval(row: dict) -> dict | None:
    question = str(row.get("question") or "").strip()
    if not question:
        return None
    return {
        key: str(row.get(key) or "").strip()
        for key in ("id", "question", "A", "B", "C", "D", "answer", "explanation")
    }


def hellaswag(row: dict) -> dict | None:
    ctx = str(row.get("ctx") or "").strip()
    if not ctx:
        return None
    return {
        "activity_label": str(row.get("activity_label") or "").strip(),
        "ctx": ctx,
        "endings": list(row.get("endings") or []),
        "label": str(row.get("label") or "").strip(),
    }


TRANSFORMS: dict[str, Transform] = {
    "alpaca": alpaca,
    "tigerbot": tigerbot,
    "gsm8k": gsm8k,
    "tinystories": tinystories,
    "wikitext": wikitext,
    "ultrafeedback": ultrafeedback,
    "chartqa": chartqa,
    "ceval": ceval,
    "hellaswag": hellaswag,
}
