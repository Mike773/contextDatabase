from typing import Any

from ..prompts import build_prompt
from .base import Analyzer

ENTITY_TYPE = "algorithm"


def _build_extraction_task(full_name: str) -> str:
    return (
        f"Ты анализируешь документ и выделяешь АЛГОРИТМЫ, процессы и способы "
        f"интерпретации метрик, связанные с направлением «{full_name}».\n\n"
        "Под алгоритмом понимается:\n"
        "- Описание процесса (последовательность шагов, регламент действий).\n"
        "- Правила расчёта или интерпретации показателей (формулы, пороги, "
        "схемы оценки).\n"
        "- Способы применения метрик к оценке работы сотрудников.\n\n"
        "Это одна из самых ценных для агента-аналитика категорий: именно из "
        "алгоритмов он поймёт, КАК устроены процессы направления и КАК "
        "интерпретируются метрики.\n\n"
        "КРИТИЧНО — ПЕРЕПРОВЕРЯЙ себя. Не всё, что выглядит как процесс, — "
        "алгоритм. В частности:\n"
        "- Простое упоминание роли или метрики БЕЗ описания их применения — "
        "НЕ алгоритм.\n"
        "- Общий факт о направлении/компании — НЕ алгоритм.\n"
        "- Единичное событие без воспроизводимого регламента — обычно НЕ "
        "алгоритм (хотя регулярный процесс — да).\n"
        "Если сомневаешься — в шагах рассуждения ЯВНО объясни себе, почему "
        "этот кандидат всё-таки алгоритм, или отбрось его.\n\n"
        "Алгоритм может быть:\n"
        "- Привязан к конкретным ролям (кто его выполняет / применяет).\n"
        "- Привязан к конкретным метрикам (что рассчитывается или "
        "интерпретируется).\n"
        "- Общим (без явной привязки к роли/метрике).\n\n"
        "Для каждого алгоритма обязательна `quote` — короткая цитата из "
        "документа, подтверждающая наличие алгоритма.\n\n"
        "ОБЯЗАТЕЛЬНО проверяй заголовок документа — иногда он прямо называет "
        "процесс/регламент, который в нём описан.\n\n"
        "КРИТИЧНО: не выдумывай расшифровки аббревиатур. Используй только те, "
        "что перечислены в «# Контекст». Если встретил незнакомую — так и "
        "напиши, что расшифровку не знаешь.\n\n"
        "Если алгоритмов в документе нет — верни пустой массив.\n\n"
        "Сначала выпиши рассуждения по шагам, в самом конце — финальный JSON "
        "СТРОГО формата:\n"
        '{"algorithms": [{"name": "...", "description": "...", '
        '"role_names": ["...", ...], "metric_names": ["...", ...], '
        '"quote": "..."}, ...], "reasoning": "..."}\n'
        "Никаких комментариев после JSON."
    )


_EXTRACTION_STEPS = [
    "Прочитай «# Контекст»: название направления, общую инфу, аббревиатуры, "
    "саммари документа, известные роли / метрики / алгоритмы направления. "
    "Выпиши для себя, что уже есть в базе.",
    "Прочитай документ целиком, включая заголовок. В 2–3 предложениях "
    "опиши, о чём он. Отметь: не является ли сам документ регламентом/"
    "алгоритмом, указанным в заголовке (например, «Регламент расчёта "
    "премии» = сам документ — алгоритм).",
    "Найди в документе все аббревиатуры. Для каждой ответь: есть ли она в "
    "списке из «# Контекст». Если нет — пиши строго «аббревиатура X не в "
    "списке, расшифровку не выдумываю».",
    "ПРЕДВАРИТЕЛЬНЫЙ список кандидатов. Выпиши всё, что может выглядеть "
    "алгоритмом/процессом/способом интерпретации метрик: последовательности "
    "шагов, регламенты, формулы, правила оценки, пороги, условия применения. "
    "Каждого кандидата — с короткой цитатой и рабочим названием. Пока не "
    "отбрасывай.",
    "РЕВИЗИЯ. Пройди по списку кандидатов из шага 4 и по КАЖДОМУ отдельно "
    "ответь на три вопроса:\n"
    "  а) Это действительно алгоритм/процесс/интерпретация, или просто "
    "упоминание?\n"
    "  б) Есть ли в документе описание ЧТО он делает или КАК (шаги, "
    "формула, условия)?\n"
    "  в) Воспроизводим ли он (регламент/формула), а не единичное событие?\n"
    "Если хотя бы один ответ «нет» — отбрасывай кандидата с пояснением. "
    "Оставляй только те, что прошли проверку.",
    "Для каждого оставшегося алгоритма составь `description`: что он "
    "делает, как работает, какие шаги / формулу / условия упоминает "
    "документ. Если в документе описаны конкретные пороги или целевые "
    "значения — включай их.",
    "Для каждого алгоритма: какие роли его выполняют / применяют? Какие "
    "метрики рассчитывает или интерпретирует? Используй канонические имена "
    "из списков «Известные роли» и «Известные метрики» из «# Контекст», "
    "если совпадают по смыслу. Если связь не очевидна — пустой массив. Не "
    "придумывай роли/метрики, которых в документе нет.",
    "Выбери одну представительную `quote` для каждого алгоритма.",
    "Напиши `reasoning` (3–4 предложения): какие алгоритмы найдены, какие "
    "кандидаты отброшены и почему.",
    "Выведи финальный JSON СТРОГО формата из «# Описания задачи». Никаких "
    "комментариев после JSON.",
]


def _format_context(
    direction: dict,
    summary: str | None,
    existing_roles: list[dict],
    existing_metrics: list[dict],
    existing_algorithms: list[dict],
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
    if existing_roles:
        lines = "\n".join(f"- {r['name']}" for r in existing_roles)
        parts.append(f"Известные роли направления:\n{lines}")
    else:
        parts.append("Известные роли направления: список пуст.")
    if existing_metrics:
        lines = "\n".join(f"- {m['name']}" for m in existing_metrics)
        parts.append(f"Известные метрики направления:\n{lines}")
    else:
        parts.append("Известные метрики направления: список пуст.")
    if existing_algorithms:
        lines = "\n".join(f"- {a['name']}" for a in existing_algorithms)
        parts.append(f"Известные алгоритмы направления:\n{lines}")
    else:
        parts.append("Известные алгоритмы направления: список пуст.")
    return "\n\n".join(parts) if parts else None


def _validate_extraction(data: Any) -> None:
    if not isinstance(data, dict):
        raise ValueError("expected object")
    algorithms = data.get("algorithms")
    if not isinstance(algorithms, list):
        raise ValueError("algorithms must be list")
    for i, a in enumerate(algorithms):
        if not isinstance(a, dict):
            raise ValueError(f"algorithms[{i}] must be object")
        for key in ("name", "description", "quote"):
            v = a.get(key)
            if not isinstance(v, str) or not v.strip():
                raise ValueError(
                    f"algorithms[{i}].{key} must be non-empty string"
                )
        for list_key in ("role_names", "metric_names"):
            lst = a.get(list_key, [])
            if not isinstance(lst, list):
                raise ValueError(f"algorithms[{i}].{list_key} must be list")
            if not all(isinstance(x, str) and x.strip() for x in lst):
                raise ValueError(
                    f"algorithms[{i}].{list_key} must contain non-empty strings"
                )
    reasoning = data.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("reasoning must be non-empty string")


class AlgorithmsAnalyzer(Analyzer):
    name = "algorithms"

    def run(self, document_id: int) -> None:
        doc = self.kb.db.fetch_document(document_id)
        if doc is None:
            raise ValueError(f"document {document_id} not found")

        direction_id = doc["direction_id"]
        direction = self.kb.db.fetch_direction(direction_id) or {}
        existing_roles = self.kb.db.fetch_roles_by_direction(direction_id)
        existing_metrics = self.kb.db.fetch_metrics_by_direction(direction_id)
        existing_algorithms = self.kb.db.fetch_algorithms_by_direction(
            direction_id
        )

        context = _format_context(
            direction,
            doc.get("summary"),
            existing_roles,
            existing_metrics,
            existing_algorithms,
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

        for alg in data["algorithms"]:
            role_names = [n.strip() for n in alg.get("role_names", [])]
            metric_names = [n.strip() for n in alg.get("metric_names", [])]
            self.kb.db.insert_extraction(
                direction_id=direction_id,
                document_id=document_id,
                entity_type=ENTITY_TYPE,
                name=alg["name"].strip(),
                description=alg["description"].strip(),
                quote=alg["quote"].strip(),
                related_role_names=role_names,
                related_metric_names=metric_names,
            )
