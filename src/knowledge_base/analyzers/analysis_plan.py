from typing import Any

from ..prompts import build_prompt
from .base import Analyzer

CRITERIA_NAMES = ("organization", "direction", "roles", "algorithms", "metrics")

_TASK = (
    "Определи, какие из 5 видов извлечения сущностей применимы к этому документу. "
    "Ключевая цель всей базы знаний — помощь агенту-аналитику оценивать работу "
    "сотрудников по метрикам; все решения фильтруй через эту линзу.\n\n"
    "КРИТИЧНО: НЕ ВЫДУМЫВАЙ расшифровки аббревиатур. Используй только те, что "
    "перечислены в разделе «# Контекст». Если встретил в документе аббревиатуру, "
    "которой там нет, — прямо напиши, что расшифровку ты не знаешь, и не пытайся "
    "её угадать. Это важно: галлюцинация расшифровки ломает дальнейший анализ.\n\n"
    "Определения 5 критериев:\n"
    "- organization — общая информация о компании (не о конкретном направлении "
    "и НЕ о процессах оценки работы сотрудников).\n"
    "- direction — общая информация о КОНКРЕТНОМ направлении (из «# Контекст»), "
    "НЕ о процессах оценки работы сотрудников этого направления.\n"
    "- roles — должности, функциональные роли, подразделения, связи (руководитель-"
    "подчинённые). Одну и ту же роль часто называют по-разному в разных документах.\n"
    "- algorithms — алгоритмы / процессы / способы интерпретации метрик (где "
    "метрика = всё, что измеряет работу сотрудника или руководителя). Могут "
    "быть привязаны к роли или общими.\n"
    "- metrics — упоминания метрик. Метрики могут быть внутри описаний "
    "алгоритмов или отдельно. Если есть описания метрик — ещё лучше.\n\n"
    "Ты обязан сначала выписать рассуждения по шагам из раздела «Структура "
    "рассуждения по шагам», и только в самом конце — выдать финальный JSON "
    "СТРОГО формата:\n"
    '{"criteria": {"organization": <bool>, "direction": <bool>, '
    '"roles": <bool>, "algorithms": <bool>, "metrics": <bool>}, '
    '"algorithms_to_run": [<имена критериев, для которых true>], '
    '"reasoning": "<3–5 предложений с обоснованием и краткими цитатами>"}\n'
    "Никаких комментариев после JSON."
)

_STEPS = [
    "Прочитай блок «# Контекст». В 1–2 предложениях своими словами выпиши: "
    "о каком направлении речь, что в общей инфе, какие аббревиатуры известны. "
    "Если есть «Саммари документа» — кратко его перескажи. Если есть «Известные "
    "роли направления» — перечисли их; если список пуст — так и напиши.",
    "Прочитай название и текст документа целиком. В 2–3 предложениях опиши "
    "своими словами, о чём документ, к чему и кому он относится.",
    "Найди в документе ВСЕ аббревиатуры (последовательности заглавных букв "
    "длиной ≥ 2 символов). Для каждой ответь: есть ли она в списке аббревиатур "
    "из «# Контекст». Если НЕТ — пиши строго: «аббревиатура X не в списке, "
    "расшифровку не выдумываю». Не пытайся угадать.",
    "Разбери документ по 5 критериям. Для КАЖДОГО критерия дай либо короткую "
    "цитату (≤ 1 предложения) + одно предложение почему он подходит, либо "
    "явное «не нашёл». По порядку: organization, direction, roles (обязательно "
    "проверь ТАКЖЕ название документа и сверь с «Известными ролями "
    "направления»), algorithms, metrics.",
    "На основе шага 4 составь список algorithms_to_run — имена критериев, "
    "для которых ответ «да». Порядок как в определении: organization, "
    "direction, roles, algorithms, metrics (без дубликатов).",
    "Составь итоговое reasoning (3–5 предложений): какие критерии включены, "
    "какие нет, и почему. По возможности вставь краткую цитату по каждому "
    "включённому критерию.",
    "После всех рассуждений выведи финальный JSON СТРОГО формата из «# "
    "Описания задачи». Никаких комментариев или текста после JSON.",
]


def _format_context(
    direction: dict,
    summary: str | None,
    roles: list[dict],
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
        parts.append(f"Аббревиатуры:\n{lines}")
    if summary and summary.strip():
        parts.append(f"Саммари документа:\n{summary.strip()}")
    if roles:
        role_lines = "\n".join(f"- {r['name']}" for r in roles)
        parts.append(f"Известные роли этого направления:\n{role_lines}")
    else:
        parts.append("Известные роли этого направления: список пуст.")
    return "\n\n".join(parts) if parts else None


def _validate(data: Any) -> None:
    if not isinstance(data, dict):
        raise ValueError("expected object")

    criteria = data.get("criteria")
    if not isinstance(criteria, dict):
        raise ValueError("criteria must be dict")
    if set(criteria.keys()) != set(CRITERIA_NAMES):
        raise ValueError(
            f"criteria keys must be exactly {sorted(CRITERIA_NAMES)}, "
            f"got {sorted(criteria)}"
        )
    for k, v in criteria.items():
        if not isinstance(v, bool):
            raise ValueError(f"criteria[{k}] must be bool, got {type(v).__name__}")

    to_run = data.get("algorithms_to_run")
    if not isinstance(to_run, list):
        raise ValueError("algorithms_to_run must be list")
    if not all(isinstance(x, str) for x in to_run):
        raise ValueError("algorithms_to_run must contain strings")
    if len(set(to_run)) != len(to_run):
        raise ValueError("algorithms_to_run must not contain duplicates")
    if not set(to_run).issubset(CRITERIA_NAMES):
        extra = set(to_run) - set(CRITERIA_NAMES)
        raise ValueError(f"algorithms_to_run has unknown names: {sorted(extra)}")
    expected = {k for k, v in criteria.items() if v}
    if set(to_run) != expected:
        raise ValueError(
            f"algorithms_to_run {sorted(to_run)} does not match "
            f"criteria-true set {sorted(expected)}"
        )

    reasoning = data.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("reasoning must be non-empty string")


class AnalysisPlanAnalyzer(Analyzer):
    name = "analysis_plan"

    def run(self, document_id: int) -> None:
        doc = self.kb.db.fetch_document(document_id)
        if doc is None:
            raise ValueError(f"document {document_id} not found")

        direction = self.kb.db.fetch_direction(doc["direction_id"]) or {}
        roles = self.kb.db.fetch_roles_by_direction(doc["direction_id"])
        context = _format_context(direction, doc.get("summary"), roles)

        prompt = build_prompt(
            task=_TASK,
            title=doc["title"],
            text=doc["text"],
            steps=_STEPS,
            context=context,
        )

        plan = self.kb.call_structured(prompt, validate=_validate)
        self.kb.db.update_document_analysis_plan(document_id, plan)
