from typing import Any

from ..prompts import build_prompt
from .base import Analyzer

_TASK = (
    "Ты анализируешь документ и готовишь по нему краткое саммари. "
    "Ты обязан сначала выписать рассуждения по шагам из раздела «Структура рассуждения по шагам», "
    "и только в самом конце — выдать финальный JSON СТРОГО формата:\n"
    '{"main_topic": "<основная тема одним словосочетанием>", '
    '"summary": "<связное саммари 3–5 предложений>", '
    '"key_themes": ["<ключевая тема 1>", "<ключевая тема 2>", ...]}\n'
    "Никаких комментариев после JSON."
)

_STEPS = [
    "Прочитай блок «# Контекст». В 1–2 предложениях выпиши своими словами, "
    "о каком направлении идёт речь и какие аббревиатуры в нём ключевые. "
    "Если блок «# Контекст» отсутствует — явно напиши: «контекст направления не задан».",
    "Прочитай документ целиком. В 2–3 предложениях опиши своими словами, "
    "о чём он и с каким направлением связан. Отдельно отметь, встречаются ли "
    "в нём аббревиатуры из шага 1 (если да — какие и где).",
    "Определи основную тему документа одним коротким словосочетанием. И напиши "
    "1–2 предложения: какие именно слова или абзацы в тексте указали тебе на "
    "эту тему — коротко процитируй их.",
    "Выпиши 3–7 ключевых тем, которые документ раскрывает. Для каждой — "
    "одно предложение обоснования: почему ты считаешь её ключевой и где она "
    "проявляется в тексте.",
    "Опираясь на рассуждения шагов 2–4, сформулируй связное саммари длиной "
    "3–5 предложений. Саммари должно опираться на факты из документа, а не на "
    "домыслы.",
    "После всех рассуждений выведи финальный JSON строго указанного формата. "
    "Никаких пояснений или текста после JSON.",
]


def _format_context(direction: dict) -> str | None:
    parts: list[str] = []
    full_name = (direction.get("full_name") or "").strip()
    if full_name:
        parts.append(f"Полное название направления: {full_name}")
    general_info = (direction.get("general_info") or "").strip()
    if general_info:
        parts.append(f"Общая информация о направлении: {general_info}")
    abbreviations = direction.get("abbreviations") or {}
    if abbreviations:
        lines = "\n".join(f"- {k}: {v}" for k, v in abbreviations.items())
        parts.append(f"Аббревиатуры:\n{lines}")
    return "\n\n".join(parts) if parts else None


def _format_summary(main_topic: str, summary: str, key_themes: list[str]) -> str:
    themes_block = "\n".join(f"- {t}" for t in key_themes)
    return f"Тема: {main_topic}\n\n{summary}\n\nКлючевые темы:\n{themes_block}"


def _validate(data: Any) -> None:
    if not isinstance(data, dict):
        raise ValueError("expected object")
    for key in ("main_topic", "summary"):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be non-empty string")
    themes = data.get("key_themes")
    if not isinstance(themes, list) or not themes:
        raise ValueError("key_themes must be non-empty list")
    if not all(isinstance(t, str) and t.strip() for t in themes):
        raise ValueError("all key_themes must be non-empty strings")


class SummaryAnalyzer(Analyzer):
    name = "summary"

    def run(self, document_id: int) -> None:
        doc = self.kb.db.fetch_document(document_id)
        if doc is None:
            raise ValueError(f"document {document_id} not found")

        direction = self.kb.db.fetch_direction(doc["direction_id"]) or {}
        context = _format_context(direction) if direction else None

        prompt = build_prompt(
            task=_TASK,
            title=doc["title"],
            text=doc["text"],
            steps=_STEPS,
            context=context,
        )

        data = self.kb.call_structured(prompt, validate=_validate)

        formatted = _format_summary(
            main_topic=data["main_topic"].strip(),
            summary=data["summary"].strip(),
            key_themes=[t.strip() for t in data["key_themes"]],
        )
        vector = self.kb.embed(formatted, direction.get("abbreviations"))
        self.kb.db.update_document_summary(document_id, formatted, vector)
