from typing import Any

from ..prompts import build_prompt
from .base import Analyzer

ENTITY_TYPE = "role"


def _build_extraction_task(full_name: str) -> str:
    return (
        f"Ты анализируешь документ и выделяешь из него роли, встречающиеся "
        f"в направлении «{full_name}»: должности, функциональные роли "
        "сотрудников, руководителей и подчинённых, упоминания подразделений — "
        "всё, что отвечает на вопрос «кто действует в этом процессе».\n\n"
        "Ключевые правила:\n"
        "- Роль = должность или функциональная роль сотрудника / руководителя.\n"
        "- Одну и ту же роль часто называют по-разному («Начальник отдела», "
        "«НО», «руководитель отдела продаж» — одна роль). Обязательно собирай "
        "ВСЕ варианты именования, встреченные в документе.\n"
        "- Если роль совпадает по смыслу с одной из списка «Известные роли "
        "направления» в «# Контекст» — используй имя из списка БУКВА В БУКВУ "
        "как каноническое; вариант из текущего документа отправь в "
        "`alternative_names`. Это обязательное условие канонизации.\n"
        "- Если новая роль — придумай лаконичное каноническое имя.\n"
        "- В описании роли собирай ВСЁ, что про неё сказано: функции / "
        "обязанности, подразделение (если указано), связи с другими ролями "
        "(кто ей руководит, кем она руководит, с кем взаимодействует).\n\n"
        "НЕ извлекай здесь:\n"
        "- Метрики, KPI, показатели, целевые значения — даже если они "
        "упомянуты рядом с ролью.\n"
        "- Алгоритмы и процессы оценки работы сотрудников.\n"
        "- Сведения о компании или о направлении в целом.\n\n"
        "КРИТИЧНО: не выдумывай расшифровки аббревиатур. Используй только "
        "те, что перечислены в «# Контекст». Если встретил незнакомую — так "
        "и напиши, что расшифровку не знаешь.\n\n"
        "Для каждой роли обязательна `quote` — короткая цитата из документа.\n\n"
        "Если ролей в документе нет — верни пустой массив.\n\n"
        "Ты обязан сначала выписать рассуждения по шагам, и только в самом "
        "конце выдать финальный JSON СТРОГО формата:\n"
        '{"roles": [{"name": "...", "alternative_names": ["...", ...], '
        '"description": "...", "quote": "..."}, ...], "reasoning": "..."}\n'
        "Никаких комментариев после JSON."
    )


_EXTRACTION_STEPS = [
    "Прочитай «# Контекст»: название направления, общую инфу, аббревиатуры, "
    "саммари документа и список известных ролей направления (имя + описание). "
    "Выпиши для себя, какие роли уже есть в базе.",
    "Прочитай документ целиком. В 2–3 предложениях опиши, о чём он.",
    "Найди в документе все аббревиатуры. Для каждой ответь: есть ли она в "
    "списке из «# Контекст». Если нет — пиши строго «аббревиатура X не в "
    "списке, расшифровку не выдумываю».",
    "Выпиши ВСЕ упоминания должностей, функциональных ролей, руководителей "
    "и подчинённых, подразделений. ОБЯЗАТЕЛЬНО отдельно проверь «Название» "
    "документа — часто заголовок прямо указывает, для какой роли написан "
    "документ (например, «Должностная инструкция менеджера по продажам» — "
    "роль «менеджер по продажам»; «Регламент работы финансового директора» "
    "— роль «финансовый директор»). Для каждого упоминания — короткая "
    "цитата (для роли из заголовка — процитируй сам заголовок) и имя "
    "роли как оно написано. Ничего пока не сворачивай.",
    "Сгруппируй варианты: если в шаге 4 одна и та же роль встречается под "
    "разными именами («Начальник отдела», «НО», «руководитель отдела "
    "продаж») — объедини их в одну группу. Для каждой группы напиши список "
    "всех вариантов. ВАЖНО: каждое конкретное упоминание роли из шага 4 "
    "должно попасть РОВНО В ОДНУ группу — не дублируй один и тот же вариант "
    "в нескольких группах. Если сомневаешься, отдельная это роль или синоним "
    "другой, выбери один из двух вариантов и зафиксируй его.",
    "Для каждой группы: сравни по смыслу со списком «Известные роли "
    "направления». Если совпадает с одной из известных — возьми её имя из "
    "списка БУКВА В БУКВУ как `name`, а все варианты из документа сложи в "
    "`alternative_names`. Если не совпадает — придумай лаконичное каноническое "
    "имя и также сложи все варианты из документа в `alternative_names`.",
    "Для каждой группы составь `description`, включив: функции / обязанности "
    "роли, подразделение (если указано), связи с другими ролями (кто ей "
    "руководит, кем она руководит, с кем взаимодействует). Опирайся только "
    "на то, что сказано в документе.",
    "Выбери одну представительную `quote` для каждой роли — цитату, из "
    "которой лучше всего видно роль и её контекст.",
    "Напиши `reasoning` (2–3 предложения): какие роли найдены, какие из них "
    "совпали с существующими, какие новые.",
    "Выведи финальный JSON СТРОГО формата из «# Описания задачи». Никаких "
    "комментариев после JSON.",
]


def _format_context(
    direction: dict,
    summary: str | None,
    existing_roles: list[dict],
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
        role_blocks = []
        for r in existing_roles:
            block = [f"- {r['name']}"]
            short = (r.get("short_description") or "").strip()
            if short:
                block.append(f"  краткое описание: {short}")
            detailed = (r.get("detailed_description") or "").strip()
            if detailed:
                block.append(f"  подробно: {detailed}")
            role_blocks.append("\n".join(block))
        parts.append(
            "Известные роли направления (используй их имена для "
            "канонизации буква в букву):\n" + "\n\n".join(role_blocks)
        )
    else:
        parts.append("Известные роли направления: список пуст.")
    return "\n\n".join(parts) if parts else None


def _validate_extraction(data: Any) -> None:
    if not isinstance(data, dict):
        raise ValueError("expected object")
    roles = data.get("roles")
    if not isinstance(roles, list):
        raise ValueError("roles must be list")
    for i, r in enumerate(roles):
        if not isinstance(r, dict):
            raise ValueError(f"roles[{i}] must be object")
        name = r.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"roles[{i}].name must be non-empty string")
        alt = r.get("alternative_names", [])
        if not isinstance(alt, list):
            raise ValueError(f"roles[{i}].alternative_names must be list")
        if not all(isinstance(x, str) and x.strip() for x in alt):
            raise ValueError(
                f"roles[{i}].alternative_names must contain non-empty strings"
            )
        description = r.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"roles[{i}].description must be non-empty string")
        quote = r.get("quote")
        if not isinstance(quote, str) or not quote.strip():
            raise ValueError(f"roles[{i}].quote must be non-empty string")
    reasoning = data.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("reasoning must be non-empty string")


class RolesAnalyzer(Analyzer):
    name = "roles"

    def run(self, document_id: int) -> None:
        doc = self.kb.db.fetch_document(document_id)
        if doc is None:
            raise ValueError(f"document {document_id} not found")

        direction_id = doc["direction_id"]
        direction = self.kb.db.fetch_direction(direction_id) or {}
        existing_roles = self.kb.db.fetch_roles_by_direction(direction_id)

        context = _format_context(direction, doc.get("summary"), existing_roles)
        full_name = (direction.get("full_name") or "").strip() or "без названия"
        prompt = build_prompt(
            task=_build_extraction_task(full_name),
            title=doc["title"],
            text=doc["text"],
            steps=_EXTRACTION_STEPS,
            context=context,
        )
        data = self.kb.call_structured(prompt, validate=_validate_extraction)

        for role in data["roles"]:
            self.kb.db.insert_extraction(
                direction_id=direction_id,
                document_id=document_id,
                entity_type=ENTITY_TYPE,
                name=role["name"].strip(),
                description=role["description"].strip(),
                quote=role["quote"].strip(),
                alternative_names=[
                    n.strip() for n in role.get("alternative_names", [])
                ],
            )
