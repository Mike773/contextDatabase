from typing import Any

from ..prompts import build_prompt
from .base import Analyzer

ENTITY_TYPE = "role"
CLAIM_SCOPE = "role"


def _build_extraction_task(full_name: str) -> str:
    return (
        f"Ты анализируешь документ в контексте направления «{full_name}» и "
        "решаешь ДВЕ задачи одновременно:\n"
        "1. Выделяешь роли — должности, функциональные роли, руководителей и "
        "подчинённых, подразделения.\n"
        "2. Для каждой выделенной роли собираешь связанные с ней процессы и "
        "церемонии (что эта роль делает в рамках направления) с атрибутами "
        "(продолжительность, периодичность, условия).\n\n"
        "Правила по ролям:\n"
        "- Роль = должность или функциональная роль сотрудника / руководителя.\n"
        "- Одну и ту же роль часто называют по-разному («Начальник отдела», "
        "«НО», «руководитель отдела продаж» — одна роль). Собирай ВСЕ "
        "варианты именования, встреченные в документе.\n"
        "- Если роль совпадает по смыслу с одной из списка «Известные роли "
        "направления» в «# Контекст» — используй имя из списка БУКВА В БУКВУ "
        "как каноническое; вариант из текущего документа отправь в "
        "`alternative_names`.\n"
        "- Если новая роль — придумай лаконичное каноническое имя.\n"
        "- В `description` включай функции, подразделение (если указано), "
        "связи с другими ролями.\n\n"
        "Правила по role_claims:\n"
        "- Каждый role_claim описывает ОДИН процесс или церемонию, в которой "
        "участвует одна или несколько ролей (регулярное действие, событие, "
        "встреча, утверждение, отчёт и т. п.).\n"
        "- `role_names` — массив канонических имён ролей, участвующих в этом "
        "процессе. Каждое имя ОБЯЗАТЕЛЬНО должно совпадать с одним из "
        "`roles[*].name`, которые ты сам же выделил в этом документе.\n"
        "- Для каждого процесса заполни `duration` (продолжительность одного "
        "выполнения), `periodicity` (как часто происходит), `conditions` "
        "(при каких условиях запускается / в какие сроки). Если в документе "
        "чего-то явно не сказано — ставь `null` для этого поля. Не угадывай.\n"
        "- `short_description` — 1 предложение о процессе.\n"
        "- `detailed_description` — развёрнуто, понятно без контекста, НЕ "
        "повторяй те же атрибуты (они уже в отдельных полях).\n\n"
        "Что НЕ извлекать:\n"
        "- Сами метрики, KPI, показатели, целевые значения (это другая "
        "задача, не эта).\n"
        "- Детальные алгоритмы расчёта метрик.\n"
        "- Общую информацию о компании или направлении.\n\n"
        "КРИТИЧНО: не выдумывай расшифровки аббревиатур. Используй только те, "
        "что перечислены в «# Контекст». Если встретил незнакомую — так и "
        "напиши.\n\n"
        "Для каждой роли и каждого role_claim обязательна `quote` — короткая "
        "цитата из документа.\n\n"
        "Если ролей/процессов нет — возвращай пустые массивы.\n\n"
        "Сначала выпиши рассуждения по шагам, в самом конце — финальный JSON "
        "СТРОГО формата:\n"
        '{"roles": [{"name": "...", "alternative_names": ["..."], '
        '"description": "...", "quote": "..."}, ...], '
        '"role_claims": [{"role_names": ["..."], "short_description": "...", '
        '"detailed_description": "...", "duration": "..."|null, '
        '"periodicity": "..."|null, "conditions": "..."|null, '
        '"quote": "..."}, ...], "reasoning": "..."}\n'
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
    "цитата и имя роли как оно написано.",
    "Сгруппируй варианты: если в шаге 4 одна и та же роль встречается под "
    "разными именами — объедини в одну группу. Для каждой группы — список "
    "всех вариантов. ВАЖНО: каждое упоминание попадает РОВНО В ОДНУ группу.",
    "Для каждой группы: сравни по смыслу со списком «Известные роли "
    "направления». Если совпадает — возьми имя из списка БУКВА В БУКВУ как "
    "`name`, варианты из документа сложи в `alternative_names`. Если нет — "
    "придумай лаконичное каноническое имя.",
    "Для каждой роли составь `description`: функции, подразделение, связи с "
    "другими ролями. Опирайся только на то, что сказано в документе.",
    "Выбери одну представительную `quote` для каждой роли.",
    "ПЕРЕХОД К ПРОЦЕССАМ. Пройди документ ещё раз и выпиши все процессы, "
    "церемонии, регулярные активности, события — ВСЁ, в чём участвует какая-"
    "то из выделенных ролей. Для каждого пункта — короткая цитата и список "
    "ролей-участников (используй канонические имена из шагов 6–7).",
    "Для каждого процесса из шага 9 извлеки атрибуты:\n"
    "   - duration: продолжительность одного выполнения (например «1-2 дня», "
    "«5 минут»), либо null если явно не сказано;\n"
    "   - periodicity: как часто (например «ежеквартально», «ежедневно»), "
    "либо null;\n"
    "   - conditions: при каких условиях или в какие сроки (например «до 15 "
    "числа первого месяца квартала», «при перевыполнении KPI»), либо null.\n"
    "   НЕ угадывай атрибуты — если документ не говорит, ставь null.",
    "Сформируй для каждого процесса `short_description` (1 предложение) и "
    "`detailed_description` (развёрнуто). В detailed_description НЕ повторяй "
    "атрибуты — они уже в своих полях.",
    "Проверь: для каждого role_claim все имена в `role_names` должны "
    "совпадать с одним из `roles[*].name` буква в букву. Если нет — "
    "исправь (либо поправь роль в roles, либо имя в role_names).",
    "Напиши `reasoning` (3–4 предложения): какие роли и процессы найдены, "
    "какие роли совпали с существующими.",
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
    role_names_set: set[str] = set()
    for i, r in enumerate(roles):
        if not isinstance(r, dict):
            raise ValueError(f"roles[{i}] must be object")
        name = r.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"roles[{i}].name must be non-empty string")
        role_names_set.add(name.strip())
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

    role_claims = data.get("role_claims")
    if not isinstance(role_claims, list):
        raise ValueError("role_claims must be list")
    for i, c in enumerate(role_claims):
        if not isinstance(c, dict):
            raise ValueError(f"role_claims[{i}] must be object")
        rn = c.get("role_names")
        if not isinstance(rn, list) or not rn:
            raise ValueError(
                f"role_claims[{i}].role_names must be non-empty list"
            )
        for n in rn:
            if not isinstance(n, str) or not n.strip():
                raise ValueError(
                    f"role_claims[{i}].role_names must contain non-empty strings"
                )
            if n.strip() not in role_names_set:
                raise ValueError(
                    f"role_claims[{i}].role_names contains {n!r} which is not "
                    f"in roles[*].name set {sorted(role_names_set)}"
                )
        for key in ("short_description", "detailed_description", "quote"):
            v = c.get(key)
            if not isinstance(v, str) or not v.strip():
                raise ValueError(
                    f"role_claims[{i}].{key} must be non-empty string"
                )
        for key in ("duration", "periodicity", "conditions"):
            v = c.get(key)
            if v is not None and (not isinstance(v, str) or not v.strip()):
                raise ValueError(
                    f"role_claims[{i}].{key} must be non-empty string or null"
                )

    reasoning = data.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("reasoning must be non-empty string")


def _format_detailed_with_attributes(
    detailed: str,
    duration: str | None,
    periodicity: str | None,
    conditions: str | None,
) -> str:
    lines: list[str] = []
    if duration:
        lines.append(f"Продолжительность: {duration.strip()}.")
    if periodicity:
        lines.append(f"Периодичность: {periodicity.strip()}.")
    if conditions:
        lines.append(f"Условия: {conditions.strip()}.")
    if lines:
        return "\n".join(lines) + "\n\n" + detailed.strip()
    return detailed.strip()


_VERIFY_TASK = (
    "Тебе даётся новое утверждение (claim) и список кандидатов — похожих "
    "утверждений, уже существующих в базе. Определи: является ли новый claim "
    "СМЫСЛОВЫМ дубликатом одного из кандидатов.\n\n"
    "Рассуждай по шагам, в конце выдай финальный JSON СТРОГО формата:\n"
    '{"match_id": <id совпавшего кандидата или null>, "reasoning": "..."}\n'
    "Никаких комментариев после JSON."
)

_VERIFY_STEPS = [
    "Прочитай новый claim.",
    "Для каждого кандидата скажи: то же ли это по смыслу. Если да — назови id.",
    "Если совпадений нет — null. Если несколько — выбери наиболее точный.",
    "Выведи финальный JSON формата из «# Описания задачи».",
]


class RolesAnalyzer(Analyzer):
    name = "roles"

    def run(self, document_id: int) -> None:
        doc = self.kb.db.fetch_document(document_id)
        if doc is None:
            raise ValueError(f"document {document_id} not found")

        direction_id = doc["direction_id"]
        direction = self.kb.db.fetch_direction(direction_id) or {}
        abbreviations = direction.get("abbreviations") or {}
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

        for claim in data["role_claims"]:
            self._upsert_role_claim(
                claim, direction_id, document_id, abbreviations
            )

    def _upsert_role_claim(
        self,
        claim: dict,
        direction_id: int,
        document_id: int,
        abbreviations: dict,
    ) -> None:
        short_desc = claim["short_description"].strip()
        detailed_desc = _format_detailed_with_attributes(
            claim["detailed_description"],
            claim.get("duration"),
            claim.get("periodicity"),
            claim.get("conditions"),
        )
        role_names = [n.strip() for n in claim["role_names"]]

        emb = self.kb.embed(short_desc, abbreviations)
        candidates = self.kb.db.find_similar_claims(
            direction_id, CLAIM_SCOPE, emb, top_k=3, threshold=0.80
        )
        if candidates:
            match_id = self._verify_claim_match(claim, candidates)
            if match_id is not None:
                self.kb.db.append_claim_document(match_id, document_id)
                return
        self.kb.db.insert_claim(
            direction_id=direction_id,
            scope=CLAIM_SCOPE,
            short_description=short_desc,
            detailed_description=detailed_desc,
            document_id=document_id,
            short_description_embedding=emb,
            role_names=role_names,
        )

    def _verify_claim_match(
        self, new_claim: dict, candidates: list[dict]
    ) -> int | None:
        candidate_ids = {c["id"] for c in candidates}

        def _validate(data: Any) -> None:
            if not isinstance(data, dict):
                raise ValueError("expected object")
            match_id = data.get("match_id")
            if match_id is not None and not isinstance(match_id, int):
                raise ValueError("match_id must be int or null")
            if match_id is not None and match_id not in candidate_ids:
                raise ValueError(
                    f"match_id {match_id} not in {sorted(candidate_ids)}"
                )
            reasoning = data.get("reasoning")
            if not isinstance(reasoning, str) or not reasoning.strip():
                raise ValueError("reasoning must be non-empty string")

        candidates_text = "\n\n".join(
            f"Кандидат id={c['id']} (similarity={c['similarity']:.2f}):\n"
            f"  short: {c['short_description']}\n"
            f"  detailed: {c['detailed_description']}"
            for c in candidates
        )
        new_text = (
            f"short: {new_claim['short_description']}\n"
            f"detailed: {new_claim['detailed_description']}"
        )
        prompt = build_prompt(
            task=_VERIFY_TASK,
            title="Проверка дубликата утверждения",
            text=new_text,
            steps=_VERIFY_STEPS,
            context=candidates_text,
        )
        data = self.kb.call_structured(prompt, validate=_validate)
        return data["match_id"]
