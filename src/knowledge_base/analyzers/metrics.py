from typing import Any

from ..prompts import build_prompt
from .base import Analyzer

ENTITY_TYPE = "metric"


def _build_extraction_task(full_name: str) -> str:
    return (
        f"Ты анализируешь документ и выделяешь из него МЕТРИКИ и показатели, "
        f"имеющие отношение к направлению «{full_name}».\n\n"
        "Метрика — это всё, что измеряет работу сотрудника или руководителя "
        "(KPI, показатели эффективности, целевые значения, числовые "
        "характеристики результата, любые измерители).\n\n"
        "Для каждой метрики извлеки ВСЮ доступную в документе информацию:\n"
        "- Если метрика ОПИСАНА (что это за показатель, как считается, что "
        "измеряет, какие уровни/целевые значения упомянуты) — используй это "
        "описание.\n"
        "- Если метрика только УПОМИНАЕТСЯ и описания нет — опиши КОНТЕКСТ "
        "УПОТРЕБЛЕНИЯ: где в документе она встречается, кто её применяет, в "
        "каком процессе, зачем. Это тоже ценно.\n\n"
        "ОБЯЗАТЕЛЬНО проверяй:\n"
        "- Заголовок документа: иногда он прямо называет метрику или тип "
        "метрик, о которых документ (например, «Регламент расчёта NPS» → "
        "метрика «NPS»).\n"
        "- Аббревиатуры из «# Контекст»: часто метрика именуется "
        "аббревиатурой (KPI, EBITDA, NPS). Расшифровку бери из списка; "
        "незнакомые аббревиатуры НЕ выдумывай.\n\n"
        "Для каждой метрики заполни `role_names` — канонические имена ролей, "
        "которые явно связаны с этой метрикой в документе (кто её "
        "применяет / к чьей работе относится). Если связь не очевидна — "
        "пустой массив.\n\n"
        "Для каждой метрики обязательна `quote` — короткая цитата из "
        "документа.\n\n"
        "Если метрик в документе нет — верни пустой массив.\n\n"
        "Сначала выпиши рассуждения по шагам, в самом конце — финальный JSON "
        "СТРОГО формата:\n"
        '{"metrics": [{"name": "...", "description": "...", '
        '"role_names": ["...", ...], "quote": "..."}, ...], '
        '"reasoning": "..."}\n'
        "Никаких комментариев после JSON."
    )


_EXTRACTION_STEPS = [
    "Прочитай «# Контекст»: название направления, общую инфу, аббревиатуры, "
    "саммари документа и список известных метрик направления.",
    "Прочитай документ целиком, включая заголовок. В 2–3 предложениях "
    "опиши, о чём он. Особо отметь, упоминает ли заголовок какую-либо "
    "конкретную метрику.",
    "Найди в документе все аббревиатуры. Для каждой ответь: есть ли она в "
    "списке из «# Контекст». Если нет — пиши строго «аббревиатура X не в "
    "списке, расшифровку не выдумываю».",
    "Выпиши ВСЕ упоминания метрик / показателей / измерителей / KPI / "
    "целевых значений, которые относятся к направлению. Проверь и сам "
    "заголовок — он может прямо называть метрику. Для каждого упоминания — "
    "короткая цитата и имя как в документе. Ничего пока не группируй.",
    "Для каждого упоминания ответь на вопрос: метрика ОПИСАНА в документе "
    "или только УПОМЯНУТА? Опиши описание (или его отсутствие) для каждой.",
    "Если метрика описана — сформулируй `description` по описанию в "
    "документе (что это за метрика, как считается, целевые значения, "
    "уровни). Если описания нет — сформулируй `description` как КОНТЕКСТ "
    "УПОТРЕБЛЕНИЯ: где встречается, кто её применяет, в каком процессе, с "
    "какой целью. В любом случае `description` должен быть НЕПУСТЫМ.",
    "Для каждой метрики: какие роли из документа с ней связаны (кто её "
    "применяет, к чьей работе относится)? Используй канонические имена "
    "ролей. Если связь не очевидна — пустой массив. Не придумывай роли, "
    "которых в документе нет.",
    "Выбери одну представительную `quote` для каждой метрики.",
    "Напиши `reasoning` (2–3 предложения): какие метрики найдены, какие "
    "описаны, какие только упомянуты.",
    "Выведи финальный JSON СТРОГО формата из «# Описания задачи». Никаких "
    "комментариев после JSON.",
]


def _format_context(
    direction: dict,
    summary: str | None,
    existing_metrics: list[dict],
) -> str | None:
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
        parts.append(f"Аббревиатуры (только читать, не выдумывать):\n{lines}")
    if summary and summary.strip():
        parts.append(f"Саммари документа:\n{summary.strip()}")
    if existing_metrics:
        metric_blocks = []
        for m in existing_metrics:
            block = [f"- {m['name']}"]
            short = (m.get("short_description") or "").strip()
            if short:
                block.append(f"  краткое описание: {short}")
            metric_blocks.append("\n".join(block))
        parts.append(
            "Известные метрики направления:\n" + "\n".join(metric_blocks)
        )
    else:
        parts.append("Известные метрики направления: список пуст.")
    return "\n\n".join(parts) if parts else None


def _validate_extraction(data: Any) -> None:
    if not isinstance(data, dict):
        raise ValueError("expected object")
    metrics = data.get("metrics")
    if not isinstance(metrics, list):
        raise ValueError("metrics must be list")
    for i, m in enumerate(metrics):
        if not isinstance(m, dict):
            raise ValueError(f"metrics[{i}] must be object")
        name = m.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"metrics[{i}].name must be non-empty string")
        description = m.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"metrics[{i}].description must be non-empty string")
        quote = m.get("quote")
        if not isinstance(quote, str) or not quote.strip():
            raise ValueError(f"metrics[{i}].quote must be non-empty string")
        role_names = m.get("role_names", [])
        if not isinstance(role_names, list):
            raise ValueError(f"metrics[{i}].role_names must be list")
        if not all(isinstance(x, str) and x.strip() for x in role_names):
            raise ValueError(
                f"metrics[{i}].role_names must contain non-empty strings"
            )
    reasoning = data.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("reasoning must be non-empty string")


class MetricsAnalyzer(Analyzer):
    name = "metrics"

    def run(self, document_id: int) -> None:
        doc = self.kb.db.fetch_document(document_id)
        if doc is None:
            raise ValueError(f"document {document_id} not found")

        direction_id = doc["direction_id"]
        direction = self.kb.db.fetch_direction(direction_id) or {}
        existing_metrics = self.kb.db.fetch_metrics_by_direction(direction_id)

        context = _format_context(
            direction, doc.get("summary"), existing_metrics
        )
        full_name = (direction.get("full_name") or "").strip() or "без названия"
        prompt = build_prompt(
            task=_build_extraction_task(full_name),
            title=doc["title"],
            text=doc["text"],
            steps=_EXTRACTION_STEPS,
            context=context,
        )
        data = self.kb.call_structured(prompt, validate=_validate_extraction)

        for metric in data["metrics"]:
            role_names = [n.strip() for n in metric.get("role_names", [])]
            self.kb.db.insert_extraction(
                direction_id=direction_id,
                document_id=document_id,
                entity_type=ENTITY_TYPE,
                name=metric["name"].strip(),
                description=metric["description"].strip(),
                quote=metric["quote"].strip(),
                related_role_names=role_names,
            )
