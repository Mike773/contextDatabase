"""Promotion pipeline: rag_v2.extractions -> rag_v2.{roles,metrics,algorithms}.

Each pending extraction row is consumed in the order roles → metrics →
algorithms so that the later stages can resolve `related_role_names` and
`related_metric_names` into foreign-key arrays.

Dedup policy per entity type:
- roles:      exact-name (case-insensitive). Match → merge (append alt_names
              and a new quote entry). Miss → insert.
- metrics:    pgvector cosine similarity (top_k=3, threshold=0.80) + an LLM
              verification call that picks exactly one match or null.
- algorithms: same as metrics.

All LLM verification calls use ``knowledge_base.prompts.build_prompt`` with
the reasoning-first template; validators reject orphan match_id values.
"""

from typing import Any, TYPE_CHECKING

from .prompts import build_prompt

if TYPE_CHECKING:
    from .kb import KnowledgeBase


# --------------------------------------------------------------------------
# LLM verification helpers
# --------------------------------------------------------------------------

_VERIFY_TASK_METRIC = (
    "Тебе даётся новая метрика и список кандидатов — похожих метрик, "
    "уже существующих в базе. Определи: является ли новая метрика "
    "СМЫСЛОВЫМ дубликатом одного из кандидатов (не побуквенно, а по смыслу "
    "измеряемого показателя).\n\n"
    "Рассуждай по шагам, в конце выдай финальный JSON СТРОГО формата:\n"
    '{"match_id": <id совпавшего кандидата или null>, "reasoning": "..."}\n'
    "Никаких комментариев после JSON."
)

_VERIFY_TASK_ALGORITHM = (
    "Тебе даётся новый алгоритм и список кандидатов — похожих алгоритмов, "
    "уже существующих в базе. Определи: описывает ли новый алгоритм тот "
    "же процесс / правило / интерпретацию, что один из кандидатов (не "
    "побуквенно, а по смыслу).\n\n"
    "Рассуждай по шагам, в конце выдай финальный JSON СТРОГО формата:\n"
    '{"match_id": <id совпавшего кандидата или null>, "reasoning": "..."}\n'
    "Никаких комментариев после JSON."
)

_VERIFY_STEPS = [
    "Прочитай нового кандидата из «# Документ».",
    "Для каждого кандидата в «# Контекст» в одном предложении скажи: "
    "то же ли это по смыслу. Если да — назови id. Если нет — коротко почему.",
    "Итог: либо id одного кандидата, либо `null`, если ни один не подошёл.",
    "Выведи финальный JSON формата из «# Описания задачи».",
]


def _build_verify_validator(candidate_ids: set[int]):
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
    return _validate


def _verify_match(
    kb: "KnowledgeBase",
    task: str,
    new_name: str,
    new_description: str,
    candidates: list[dict],
) -> int | None:
    candidate_ids = {c["id"] for c in candidates}
    candidates_text = "\n\n".join(
        f"Кандидат id={c['id']} (similarity={c['similarity']:.2f}):\n"
        f"  name: {c['name']}\n"
        f"  description: {c['detailed_description']}"
        for c in candidates
    )
    new_text = f"name: {new_name}\ndescription: {new_description}"
    prompt = build_prompt(
        task=task,
        title="Проверка дубликата",
        text=new_text,
        steps=_VERIFY_STEPS,
        context=candidates_text,
    )
    data = kb.call_structured(
        prompt, validate=_build_verify_validator(candidate_ids)
    )
    return data["match_id"]


# --------------------------------------------------------------------------
# Promoters
# --------------------------------------------------------------------------

def promote_roles(kb: "KnowledgeBase", direction_id: int) -> dict:
    direction = kb.db.fetch_direction(direction_id) or {}
    abbreviations = direction.get("abbreviations") or {}
    pending = kb.db.fetch_pending_extractions(direction_id, "role")

    created = 0
    merged = 0
    for ex in pending:
        name = (ex["name"] or "").strip()
        if not name:
            continue
        detailed = (ex["description"] or "").strip()
        quote = (ex["quote"] or "").strip()
        alt_names = list(ex["alternative_names"] or [])

        existing_id = kb.db.find_role_by_name(direction_id, name)
        if existing_id is not None:
            kb.db.merge_role(
                existing_id,
                alternative_names=alt_names,
                document_id=ex["document_id"],
                quote=quote,
            )
            merged += 1
        else:
            name_emb = kb.embed(name, abbreviations)
            kb.db.insert_role(
                direction_id=direction_id,
                name=name,
                detailed_description=detailed,
                alternative_names=alt_names,
                document_id=ex["document_id"],
                quote=quote,
                name_embedding=name_emb,
            )
            created += 1
        kb.db.mark_extraction_loaded(ex["id"])
    return {"created": created, "merged": merged, "processed": len(pending)}


def _resolve_role_ids(
    kb: "KnowledgeBase", direction_id: int, role_names: list[str]
) -> tuple[list[int], list[str]]:
    if not role_names:
        return [], []
    resolved: list[int] = []
    unresolved: list[str] = []
    seen: set[int] = set()
    for n in role_names:
        rid = kb.db.find_role_by_any_name(direction_id, n)
        if rid is None:
            unresolved.append(n)
            continue
        if rid in seen:
            continue
        seen.add(rid)
        resolved.append(rid)
    return resolved, unresolved


def _resolve_metric_ids(
    kb: "KnowledgeBase", direction_id: int, metric_names: list[str]
) -> tuple[list[int], list[str]]:
    if not metric_names:
        return [], []
    resolved: list[int] = []
    unresolved: list[str] = []
    seen: set[int] = set()
    metrics = kb.db.fetch_metrics_by_direction(direction_id)
    lookup = {m["name"].lower(): m["id"] for m in metrics}
    for n in metric_names:
        key = n.strip().lower()
        mid = lookup.get(key)
        if mid is None:
            unresolved.append(n)
            continue
        if mid in seen:
            continue
        seen.add(mid)
        resolved.append(mid)
    return resolved, unresolved


def promote_metrics(kb: "KnowledgeBase", direction_id: int) -> dict:
    direction = kb.db.fetch_direction(direction_id) or {}
    abbreviations = direction.get("abbreviations") or {}
    pending = kb.db.fetch_pending_extractions(direction_id, "metric")

    created = 0
    merged = 0
    unresolved_roles: list[str] = []
    for ex in pending:
        name = (ex["name"] or "").strip()
        if not name:
            continue
        detailed = (ex["description"] or "").strip()
        role_ids, missing = _resolve_role_ids(
            kb, direction_id, ex["related_role_names"] or []
        )
        unresolved_roles.extend(missing)

        desc_emb = kb.embed(detailed, abbreviations)
        candidates = kb.db.find_similar_metrics(
            direction_id, desc_emb, top_k=3, threshold=0.80
        )
        if candidates:
            match_id = _verify_match(
                kb, _VERIFY_TASK_METRIC, name, detailed, candidates
            )
            if match_id is not None:
                kb.db.merge_metric(
                    match_id,
                    document_id=ex["document_id"],
                    role_ids=role_ids,
                )
                merged += 1
                kb.db.mark_extraction_loaded(ex["id"])
                continue

        name_emb = kb.embed(name, abbreviations)
        kb.db.insert_metric(
            direction_id=direction_id,
            name=name,
            detailed_description=detailed,
            document_id=ex["document_id"],
            role_ids=role_ids,
            name_embedding=name_emb,
            description_embedding=desc_emb,
        )
        created += 1
        kb.db.mark_extraction_loaded(ex["id"])

    return {
        "created": created,
        "merged": merged,
        "processed": len(pending),
        "unresolved_role_names": sorted(set(unresolved_roles)),
    }


def promote_algorithms(kb: "KnowledgeBase", direction_id: int) -> dict:
    direction = kb.db.fetch_direction(direction_id) or {}
    abbreviations = direction.get("abbreviations") or {}
    pending = kb.db.fetch_pending_extractions(direction_id, "algorithm")

    created = 0
    merged = 0
    unresolved_roles: list[str] = []
    unresolved_metrics: list[str] = []
    for ex in pending:
        name = (ex["name"] or "").strip()
        if not name:
            continue
        detailed = (ex["description"] or "").strip()
        quote = (ex["quote"] or "").strip()

        role_ids, missing_roles = _resolve_role_ids(
            kb, direction_id, ex["related_role_names"] or []
        )
        unresolved_roles.extend(missing_roles)
        metric_ids, missing_metrics = _resolve_metric_ids(
            kb, direction_id, ex["related_metric_names"] or []
        )
        unresolved_metrics.extend(missing_metrics)

        desc_emb = kb.embed(detailed, abbreviations)
        candidates = kb.db.find_similar_algorithms(
            direction_id, desc_emb, top_k=3, threshold=0.80
        )
        if candidates:
            match_id = _verify_match(
                kb, _VERIFY_TASK_ALGORITHM, name, detailed, candidates
            )
            if match_id is not None:
                kb.db.merge_algorithm(
                    match_id,
                    document_id=ex["document_id"],
                    quote=quote,
                    role_ids=role_ids,
                    metric_ids=metric_ids,
                )
                merged += 1
                kb.db.mark_extraction_loaded(ex["id"])
                continue

        name_emb = kb.embed(name, abbreviations)
        kb.db.insert_algorithm(
            direction_id=direction_id,
            name=name,
            detailed_description=detailed,
            document_id=ex["document_id"],
            quote=quote,
            role_ids=role_ids,
            metric_ids=metric_ids,
            name_embedding=name_emb,
            description_embedding=desc_emb,
        )
        created += 1
        kb.db.mark_extraction_loaded(ex["id"])

    return {
        "created": created,
        "merged": merged,
        "processed": len(pending),
        "unresolved_role_names": sorted(set(unresolved_roles)),
        "unresolved_metric_names": sorted(set(unresolved_metrics)),
    }


def promote_all(kb: "KnowledgeBase", direction_id: int) -> dict:
    return {
        "roles": promote_roles(kb, direction_id),
        "metrics": promote_metrics(kb, direction_id),
        "algorithms": promote_algorithms(kb, direction_id),
    }
