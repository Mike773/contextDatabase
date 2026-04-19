from typing import Any

from ..prompts import build_prompt
from .base import Analyzer

SCOPE = "organization"

_EXTRACTION_TASK = (
    "Ты анализируешь документ и выделяешь из него общую информацию о КОМПАНИИ/"
    "ОРГАНИЗАЦИИ — такую, которая НЕ относится к конкретному направлению и НЕ "
    "относится к процессам оценки работы сотрудников.\n\n"
    "Цель всей базы знаний — помощь агенту-аналитику оценивать работу сотрудников "
    "по метрикам. Сам этот анализатор метрики и роли НЕ извлекает — он нужен "
    "только для общего корпоративного контекста.\n\n"
    "КРИТИЧНО: не выдумывай расшифровки аббревиатур. Используй только те, что "
    "перечислены в «# Контекст». Если встретил незнакомую — так и напиши.\n\n"
    "Типы сущностей на извлечение:\n"
    "- terms — термины / понятия / наименования организационного уровня "
    "(названия систем, департаментов верхнего уровня, корпоративные понятия).\n"
    "- claims — утверждения о компании (факты, положения, правила), которые "
    "касаются организации в целом.\n\n"
    "Для каждого извлечённого элемента ОБЯЗАТЕЛЬНА `quote` — короткая цитата "
    "из документа, подтверждающая, что это там действительно сказано.\n\n"
    "Если в документе нет org-level контента — верни пустые массивы. Это "
    "валидный и ожидаемый случай для документов про направление или оценку "
    "сотрудников.\n\n"
    "Ты обязан сначала выписать рассуждения по шагам, и только в самом конце "
    "выдать финальный JSON СТРОГО формата:\n"
    '{"terms": [{"name": "...", "short_description": "...", '
    '"detailed_description": "...", "quote": "..."}, ...], '
    '"claims": [{"short_description": "...", "detailed_description": "...", '
    '"quote": "..."}, ...], "reasoning": "..."}\n'
    "Никаких комментариев после JSON."
)

_EXTRACTION_STEPS = [
    "Прочитай «# Контекст»: название направления, общую инфу, аббревиатуры, "
    "саммари документа и известные org-level термины.",
    "Прочитай документ целиком. В 2–3 предложениях своими словами опиши, "
    "о чём он.",
    "Найди в документе все аббревиатуры. Для каждой ответь: есть ли в списке "
    "из «# Контекст». Если нет — пиши строго «аббревиатура X не в списке, "
    "расшифровку не выдумываю».",
    "Пройдись по документу и выпиши всё, что относится к КОМПАНИИ В ЦЕЛОМ "
    "(не к направлению и не к оценке сотрудников). Для каждого пункта — "
    "цитата + одно предложение, почему это org-level. Если ничего нет — "
    "явно напиши «общей информации о компании не нашёл».",
    "Из шага 4 оформи terms (существительные / понятия / названия): "
    "для каждого `name`, `short_description`, `detailed_description`, "
    "`quote`. Сверяйся со списком «Известные org-level термины» — если такой "
    "же name уже есть, всё равно приведи его снова с новой цитатой, дедуп "
    "сделает код. Пустой массив — валидно.",
    "Из шага 4 оформи claims (утверждения-предложения): `short_description` "
    "(1 предложение), `detailed_description` (развёрнуто и понятно без "
    "контекста!), `quote`. Пустой массив — валидно.",
    "Напиши `reasoning` (2–3 предложения): что найдено, что нет и почему.",
    "Выведи финальный JSON СТРОГО формата из «# Описания задачи». Никаких "
    "комментариев после JSON.",
]


_VERIFY_TASK = (
    "Тебе даётся новое утверждение (claim) и список кандидатов — похожих "
    "утверждений, уже существующих в базе. Определи: является ли новый claim "
    "СМЫСЛОВЫМ дубликатом какого-либо из кандидатов (не побуквенно, а по "
    "смыслу).\n\n"
    "Рассуждай по шагам и в конце выдай финальный JSON СТРОГО формата:\n"
    '{"match_id": <id совпавшего кандидата или null>, "reasoning": "..."}\n'
    "Никаких комментариев после JSON."
)

_VERIFY_STEPS = [
    "Прочитай новый claim (short + detailed) из «# Документ».",
    "Для каждого кандидата в «# Контекст» в одном предложении скажи: "
    "то же ли это утверждение по смыслу. Если да — назови id кандидата "
    "и короткое обоснование. Если нет — скажи, почему это другое.",
    "Решение: если подходит один — укажи его id. Если подходят несколько — "
    "выбери наиболее точный. Если ни один не подошёл — `null`.",
    "Выведи финальный JSON формата из «# Описания задачи».",
]


def _format_context(
    direction: dict,
    summary: str | None,
    existing_terms: list[dict],
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
    if existing_terms:
        term_lines = "\n".join(f"- {t['name']}" for t in existing_terms)
        parts.append(
            f"Известные org-level термины этого направления:\n{term_lines}"
        )
    else:
        parts.append(
            "Известные org-level термины этого направления: список пуст."
        )
    return "\n\n".join(parts) if parts else None


def _validate_extraction(data: Any) -> None:
    if not isinstance(data, dict):
        raise ValueError("expected object")
    for key in ("terms", "claims"):
        if not isinstance(data.get(key), list):
            raise ValueError(f"{key} must be list")
    for i, t in enumerate(data["terms"]):
        if not isinstance(t, dict):
            raise ValueError(f"terms[{i}] must be object")
        name = t.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"terms[{i}].name must be non-empty string")
        for sub in ("short_description", "detailed_description"):
            if sub in t and t[sub] is not None and not isinstance(t[sub], str):
                raise ValueError(f"terms[{i}].{sub} must be string or null")
        quote = t.get("quote")
        if not isinstance(quote, str) or not quote.strip():
            raise ValueError(f"terms[{i}].quote must be non-empty string")
    for i, c in enumerate(data["claims"]):
        if not isinstance(c, dict):
            raise ValueError(f"claims[{i}] must be object")
        for key in ("short_description", "detailed_description"):
            value = c.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"claims[{i}].{key} must be non-empty string")
        quote = c.get("quote")
        if not isinstance(quote, str) or not quote.strip():
            raise ValueError(f"claims[{i}].quote must be non-empty string")
    reasoning = data.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("reasoning must be non-empty string")


class OrganizationAnalyzer(Analyzer):
    name = "organization"

    def run(self, document_id: int) -> None:
        doc = self.kb.db.fetch_document(document_id)
        if doc is None:
            raise ValueError(f"document {document_id} not found")

        direction_id = doc["direction_id"]
        direction = self.kb.db.fetch_direction(direction_id) or {}
        abbreviations = direction.get("abbreviations") or {}
        existing_terms = self.kb.db.fetch_terms_by_scope(direction_id, SCOPE)

        context = _format_context(direction, doc.get("summary"), existing_terms)
        prompt = build_prompt(
            task=_EXTRACTION_TASK,
            title=doc["title"],
            text=doc["text"],
            steps=_EXTRACTION_STEPS,
            context=context,
        )
        data = self.kb.call_structured(prompt, validate=_validate_extraction)

        for t in data["terms"]:
            self._upsert_term(t, direction_id, document_id, abbreviations)

        for c in data["claims"]:
            self._upsert_claim(c, direction_id, document_id, abbreviations)

    def _upsert_term(
        self,
        term: dict,
        direction_id: int,
        document_id: int,
        abbreviations: dict,
    ) -> None:
        name = term["name"].strip()
        quote = term["quote"].strip()
        existing_id = self.kb.db.find_term_by_name(direction_id, SCOPE, name)
        if existing_id is not None:
            self.kb.db.append_term_quote(existing_id, document_id, quote)
            return
        short_desc = (term.get("short_description") or "").strip()
        detailed_desc = (term.get("detailed_description") or "").strip()
        name_emb = self.kb.embed(name, abbreviations)
        short_emb = (
            self.kb.embed(short_desc, abbreviations) if short_desc else None
        )
        self.kb.db.insert_term(
            direction_id=direction_id,
            scope=SCOPE,
            name=name,
            short_description=short_desc or None,
            detailed_description=detailed_desc or None,
            document_id=document_id,
            quote=quote,
            name_embedding=name_emb,
            short_description_embedding=short_emb,
        )

    def _upsert_claim(
        self,
        claim: dict,
        direction_id: int,
        document_id: int,
        abbreviations: dict,
    ) -> None:
        short_desc = claim["short_description"].strip()
        detailed_desc = claim["detailed_description"].strip()
        emb = self.kb.embed(short_desc, abbreviations)
        candidates = self.kb.db.find_similar_claims(
            direction_id, SCOPE, emb, top_k=3, threshold=0.80
        )
        if candidates:
            match_id = self._verify_claim_match(claim, candidates)
            if match_id is not None:
                self.kb.db.append_claim_document(match_id, document_id)
                return
        self.kb.db.insert_claim(
            direction_id=direction_id,
            scope=SCOPE,
            short_description=short_desc,
            detailed_description=detailed_desc,
            document_id=document_id,
            short_description_embedding=emb,
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
