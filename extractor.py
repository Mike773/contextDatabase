"""Выборка контекста из базы знаний rag_v2 под инструкцию аналитика.

Самодостаточный модуль: ничего не импортирует из остального
knowledge_base-пакета. Внутри — один класс KnowledgeBaseExtractor,
получающий на вход функции llm_call, embed и строку подключения к БД.

Промпты фреймят задачу как ОТБОР записей из базы знаний; LLM не
отвечает на инструкцию аналитика. Если по какому-либо разделу
ничего не подошло, возвращается строка NOT_FOUND.
"""

import json
import re
from typing import Any, Callable

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector


class KnowledgeBaseExtractor:
    NOT_FOUND = "в базе знаний я не нашел данной информации"
    TOP_K = 10
    CLAIMS_SIM_THRESHOLD = 0.60
    TERMS_SIM_THRESHOLD = 0.55
    METRICS_SIM_THRESHOLD = 0.55
    ALGORITHMS_SIM_THRESHOLD = 0.55
    MAX_LLM_RETRIES = 3
    ROLE_CANDIDATES_CAP = 15

    def __init__(
        self,
        *,
        llm_call: Callable[[str], str],
        embed: Callable[[str], list[float]],
        db_connection_string: str,
    ) -> None:
        self._llm_call = llm_call
        self._embed_fn = embed
        self._dsn = db_connection_string

    # ==================================================================
    # Public API
    # ==================================================================

    def extract(self, direction_id: int, instruction: str) -> dict:
        conn = psycopg2.connect(self._dsn)
        conn.autocommit = True
        register_vector(conn)
        try:
            with conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor
            ) as cur:
                direction = self._fetch_direction(cur, direction_id)
                if direction is None:
                    raise ValueError(f"direction {direction_id} not found")

                abbreviations = direction.get("abbreviations") or {}

                roles = self._fetch_roles(cur, direction_id)
                org_terms = self._fetch_terms(cur, direction_id, "organization")
                org_claims = self._fetch_claims_full(
                    cur, direction_id, "organization"
                )
                dir_terms = self._fetch_terms(cur, direction_id, "direction")
                dir_claims = self._fetch_claims_full(
                    cur, direction_id, "direction"
                )

                instruction_embedding = self._embed_for_retrieval(
                    instruction, abbreviations
                )

                matched_roles = self._match_roles(roles, instruction, direction)
                matched_role_ids = [r["id"] for r in matched_roles]
                matched_role_names = [r["name"] for r in matched_roles]

                role_claims = self._select_role_claims(
                    cur,
                    direction_id=direction_id,
                    matched_roles=matched_roles,
                    matched_role_names=matched_role_names,
                    instruction=instruction,
                    instruction_embedding=instruction_embedding,
                    direction=direction,
                )
                role_terms = self._select_role_terms(
                    cur,
                    direction_id=direction_id,
                    matched_roles=matched_roles,
                    instruction=instruction,
                    instruction_embedding=instruction_embedding,
                    direction=direction,
                )
                metrics = self._select_metrics(
                    cur,
                    direction_id=direction_id,
                    matched_roles=matched_roles,
                    matched_role_ids=matched_role_ids,
                    instruction=instruction,
                    instruction_embedding=instruction_embedding,
                    direction=direction,
                )
                algorithms = self._select_algorithms(
                    cur,
                    direction_id=direction_id,
                    matched_roles=matched_roles,
                    matched_role_ids=matched_role_ids,
                    instruction=instruction,
                    instruction_embedding=instruction_embedding,
                    direction=direction,
                )
        finally:
            conn.close()

        if org_terms or org_claims:
            organization_description: Any = {
                "terms": org_terms,
                "claims": org_claims,
            }
        else:
            organization_description = self.NOT_FOUND

        direction_description = {
            "direction": {
                "id": direction["id"],
                "short_name": direction["short_name"],
                "full_name": direction["full_name"],
                "general_info": direction.get("general_info") or "",
            },
            "terms": dir_terms,
            "claims": dir_claims,
        }

        if role_claims or role_terms:
            role_related: Any = {
                "matched_roles": [
                    {
                        "id": r["id"],
                        "name": r["name"],
                        "short_description": r.get("short_description"),
                    }
                    for r in matched_roles
                ],
                "claims": role_claims,
                "terms": role_terms,
            }
        else:
            role_related = self.NOT_FOUND

        return {
            "organization_description": organization_description,
            "direction_description": direction_description,
            "role_related": role_related,
            "metrics": metrics or self.NOT_FOUND,
            "algorithms": algorithms or self.NOT_FOUND,
        }

    # ==================================================================
    # SQL
    # ==================================================================

    @staticmethod
    def _fetch_direction(cur, direction_id: int) -> dict | None:
        cur.execute(
            "SELECT id, short_name, full_name, general_info, abbreviations "
            "FROM rag_v2.directions WHERE id = %s",
            (direction_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    @staticmethod
    def _fetch_roles(cur, direction_id: int) -> list[dict]:
        cur.execute(
            "SELECT id, name, short_description, detailed_description "
            "FROM rag_v2.roles WHERE direction_id = %s ORDER BY id",
            (direction_id,),
        )
        return [dict(r) for r in cur.fetchall()]

    @staticmethod
    def _fetch_terms(cur, direction_id: int, scope: str) -> list[dict]:
        cur.execute(
            "SELECT id, name, short_description, detailed_description "
            "FROM rag_v2.terms "
            "WHERE direction_id = %s AND scope = %s ORDER BY id",
            (direction_id, scope),
        )
        return [dict(r) for r in cur.fetchall()]

    @staticmethod
    def _fetch_claims_full(cur, direction_id: int, scope: str) -> list[dict]:
        cur.execute(
            "SELECT id, short_description, detailed_description "
            "FROM rag_v2.claims "
            "WHERE direction_id = %s AND scope = %s ORDER BY id",
            (direction_id, scope),
        )
        return [dict(r) for r in cur.fetchall()]

    # ==================================================================
    # Embedding
    # ==================================================================

    @staticmethod
    def _expand_abbreviations(text: str, abbreviations: dict) -> str:
        if not abbreviations:
            return text
        for key in sorted(abbreviations.keys(), key=len, reverse=True):
            text = re.sub(
                rf"\b{re.escape(key)}\b", str(abbreviations[key]), text
            )
        return text

    def _embed_for_retrieval(
        self, text: str, abbreviations: dict
    ) -> list[float]:
        return self._embed_fn(self._expand_abbreviations(text, abbreviations))

    # ==================================================================
    # LLM
    # ==================================================================

    def _call_structured(
        self,
        prompt: str,
        *,
        validate: Callable[[Any], None],
    ) -> Any:
        last_err: Exception | None = None
        current_prompt = prompt
        for attempt in range(self.MAX_LLM_RETRIES):
            response = self._llm_call(current_prompt)
            try:
                data = _extract_json(response)
                validate(data)
                return data
            except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
                last_err = e
                current_prompt = (
                    prompt
                    + f"\n\n# Предыдущая попытка {attempt + 1} не дала валидный JSON: {e}.\n"
                    "В новой попытке ОБЯЗАТЕЛЬНО сохрани рассуждения по шагам, "
                    "но в самом конце после последнего шага выведи ровно один "
                    "валидный JSON-блок, строго соответствующий формату из «Описания задачи»."
                )
        raise RuntimeError(
            f"structured LLM call failed after {self.MAX_LLM_RETRIES} attempts: {last_err}"
        )

    @staticmethod
    def _build_prompt(
        *,
        task: str,
        title: str,
        text: str,
        steps: list[str],
        context: str,
    ) -> str:
        numbered = "\n".join(f"Шаг {i + 1}. {s}" for i, s in enumerate(steps))
        blocks: list[str] = []
        if context.strip():
            blocks.append(f"# Контекст\n{context.strip()}")
        blocks.append(f"# Описание задачи\n{task}")
        blocks.append(f"# Документ\nНазвание: {title}\n\n{text}")
        blocks.append(f"# Структура рассуждения по шагам\n{numbered}")
        blocks.append(
            "# Начни с шага 1\n"
            f"Выполни ВСЕ {len(steps)} шагов последовательно, каждый начинай "
            "с новой строки вида «Шаг N.». Не пропускай шаги и не сокращай "
            "рассуждения. Только после последнего шага выдай финальный "
            "ответ, указанный в «Описании задачи».\n\n"
            "Шаг 1."
        )
        return "\n\n".join(blocks)

    # ==================================================================
    # Prompt helpers
    # ==================================================================

    @staticmethod
    def _format_abbreviations(abbreviations: dict) -> str:
        if not abbreviations:
            return "— аббревиатуры не заданы —"
        return "\n".join(
            f"- {k} — {v}"
            for k, v in sorted(abbreviations.items(), key=lambda kv: kv[0])
        )

    def _format_direction_header(self, direction: dict) -> str:
        short_name = direction.get("short_name") or ""
        full_name = direction.get("full_name") or ""
        general_info = (direction.get("general_info") or "").strip() or "— не заполнено —"
        abbr_list = self._format_abbreviations(
            direction.get("abbreviations") or {}
        )
        return (
            f"Направление: {full_name} (короткое имя: {short_name})\n"
            f"Общая информация о направлении:\n{general_info}\n\n"
            "Аббревиатуры направления (использовать ТОЛЬКО из этого списка):\n"
            f"{abbr_list}"
        )

    @staticmethod
    def _trim(value: Any, limit: int = 400) -> str:
        if value is None:
            return ""
        text = str(value).replace("\n", " ").strip()
        if len(text) <= limit:
            return text
        return text[: limit - 1] + "…"

    # ==================================================================
    # Phase C — role matching
    # ==================================================================

    def _match_roles(
        self, roles: list[dict], instruction: str, direction: dict
    ) -> list[dict]:
        if not roles:
            return []

        role_lines = "\n".join(
            f"id={r['id']} | name={r['name']} | "
            f"short_description={self._trim(r.get('short_description'))}"
            for r in roles
        )
        context = (
            f"{self._format_direction_header(direction)}\n\n"
            f"# Список ролей направления\n{role_lines}"
        )

        task = (
            "Твоя задача — ВЫБРАТЬ из приведённого списка ролей направления те,\n"
            "которые описывает постановка задачи аналитика. Ты НЕ отвечаешь\n"
            "на саму задачу и НЕ выполняешь её. Ты только отбираешь записи\n"
            "из базы знаний.\n\n"
            "Правила:\n"
            "- Роль может называться по-разному (синонимы, варианты\n"
            "  именования, аббревиатуры) — опирайся на смысл, не на\n"
            "  побуквенное совпадение.\n"
            "- Может подойти несколько ролей сразу — это нормально.\n"
            "- Может не подойти ни одной — это тоже нормально, верни\n"
            "  пустой список.\n"
            "- НЕ добавляй роли, id которых отсутствуют в списке «# Контекст».\n"
            "- НЕ выдумывай новые роли.\n"
            "- Аббревиатуры расшифровывай ТОЛЬКО через список из «# Контекст».\n\n"
            "Финальный ответ СТРОГО формата:\n"
            '{"role_ids": [<int>, ...], "reasoning": "<1–2 предложения>"}\n'
            "Никаких комментариев после JSON."
        )

        steps = [
            "Выпиши список всех ролей из «# Контекст» — id и name.",
            "Прочитай задачу аналитика в «# Документ». В 1–2 предложениях опиши, о какой должности/роли/функции идёт речь.",
            "Разверни в задаче аббревиатуры из «# Контекст» (для себя, в рассуждениях). Незнакомые не расшифровывай.",
            "Для КАЖДОЙ роли из списка отдельно дай вердикт — одно предложение «подходит, потому что ...» или «не подходит, потому что ...».",
            "Собери id всех ролей, отмеченных в шаге 4 как подходящие. Если ни одна — пустой список.",
            "Напиши reasoning — 1–2 предложения про итог выбора.",
            "Выведи финальный JSON строго по формату из «# Описания задачи».",
        ]

        prompt = self._build_prompt(
            task=task,
            title="Подбор ролей из базы знаний",
            text=f"Задача аналитика:\n{instruction}",
            steps=steps,
            context=context,
        )

        id_to_role = {r["id"]: r for r in roles}
        data = self._call_structured(
            prompt, validate=self._ids_validator("role_ids", id_to_role)
        )
        return [
            id_to_role[i]
            for i in dict.fromkeys(data["role_ids"])
            if i in id_to_role
        ]

    # ==================================================================
    # Phase D — selectors
    # ==================================================================

    def _select_role_claims(
        self,
        cur,
        *,
        direction_id: int,
        matched_roles: list[dict],
        matched_role_names: list[str],
        instruction: str,
        instruction_embedding: list[float],
        direction: dict,
    ) -> list[dict]:
        candidates_by_id: dict[int, dict] = {}

        if matched_role_names:
            cur.execute(
                "SELECT id, scope, short_description, detailed_description, "
                "       role_names "
                "FROM rag_v2.claims "
                "WHERE direction_id = %s AND role_names && %s::text[] "
                "ORDER BY id",
                (direction_id, matched_role_names),
            )
            for r in cur.fetchall():
                candidates_by_id[r["id"]] = dict(r)

        cur.execute(
            "SELECT id, scope, short_description, detailed_description, "
            "       role_names, "
            "       1 - (short_description_embedding <=> %s::vector) AS similarity "
            "FROM rag_v2.claims "
            "WHERE direction_id = %s AND scope = 'role' "
            "  AND short_description_embedding IS NOT NULL "
            "ORDER BY short_description_embedding <=> %s::vector "
            "LIMIT %s",
            (
                instruction_embedding,
                direction_id,
                instruction_embedding,
                self.TOP_K,
            ),
        )
        for r in cur.fetchall():
            row = dict(r)
            if row.pop("similarity") < self.CLAIMS_SIM_THRESHOLD:
                continue
            candidates_by_id.setdefault(row["id"], row)

        if not candidates_by_id:
            return []

        candidates = list(candidates_by_id.values())[: self.ROLE_CANDIDATES_CAP]
        id_to_cand = {c["id"]: c for c in candidates}

        candidate_lines = "\n".join(
            f"id={c['id']} | scope={c.get('scope', '')} | "
            f"role_names={list(c.get('role_names') or [])} | "
            f"short={self._trim(c.get('short_description'))} | "
            f"detailed={self._trim(c.get('detailed_description'))}"
            for c in candidates
        )
        context_parts = [self._format_direction_header(direction)]
        if matched_roles:
            context_parts.append(
                "# Выбранные роли\n"
                + "\n".join(f"- {r['name']}" for r in matched_roles)
            )
        context_parts.append("# Кандидаты-утверждения\n" + candidate_lines)

        task = (
            "Твоя задача — ВЫБРАТЬ из приведённого списка утверждений-кандидатов\n"
            "те, которые относятся к задаче аналитика и/или к выбранным ролям.\n"
            "Ты НЕ отвечаешь на задачу, НЕ формулируешь новых утверждений.\n"
            "Ты только отбираешь записи из базы знаний, УЖЕ присутствующие\n"
            "в списке «# Контекст».\n\n"
            "Утверждение подходит, если:\n"
            "- оно прямо упоминает выбранную роль (через role_names или по\n"
            "  смыслу), ИЛИ\n"
            "- описывает процесс/ограничение/правило, о котором спрашивает\n"
            "  задача аналитика применительно к этой роли.\n\n"
            "НЕ включай утверждения, лишь косвенно пересекающиеся с темой.\n"
            "НЕ добавляй id, отсутствующие в списке. НЕ выдумывай новые\n"
            "утверждения. Если ни одно не подходит — верни пустой список.\n"
            "Это валидный ответ.\n\n"
            "Аббревиатуры расшифровывай ТОЛЬКО через список из «# Контекст».\n\n"
            "Финальный ответ СТРОГО формата:\n"
            '{"claim_ids": [<int>, ...], "reasoning": "<1–2 предложения>"}\n'
            "Никаких комментариев после JSON."
        )

        steps = [
            "Перечитай задачу аналитика. В 1–2 предложениях выдели: про какую роль, какой процесс, какой аспект.",
            "Разверни аббревиатуры из «# Контекст» (для себя, в рассуждениях).",
            "Пройди по КАЖДОМУ кандидату отдельно. По каждому дай одно предложение: «подходит, потому что ...» или «не подходит, потому что ...».",
            "Собери id подходящих. Если ни одного — пустой список.",
            "Напиши reasoning — 1–2 предложения про принцип отбора.",
            "Выведи финальный JSON строго по формату из «# Описания задачи».",
        ]

        prompt = self._build_prompt(
            task=task,
            title="Выборка утверждений, упоминающих роль",
            text=f"Задача аналитика:\n{instruction}",
            steps=steps,
            context="\n\n".join(context_parts),
        )

        data = self._call_structured(
            prompt, validate=self._ids_validator("claim_ids", id_to_cand)
        )
        return [
            id_to_cand[i]
            for i in dict.fromkeys(data["claim_ids"])
            if i in id_to_cand
        ]

    def _select_role_terms(
        self,
        cur,
        *,
        direction_id: int,
        matched_roles: list[dict],
        instruction: str,
        instruction_embedding: list[float],
        direction: dict,
    ) -> list[dict]:
        cur.execute(
            "SELECT id, scope, name, short_description, detailed_description, "
            "       1 - (COALESCE(short_description_embedding, name_embedding) <=> %s::vector) AS similarity "
            "FROM rag_v2.terms "
            "WHERE direction_id = %s "
            "  AND (short_description_embedding IS NOT NULL "
            "       OR name_embedding IS NOT NULL) "
            "ORDER BY COALESCE(short_description_embedding, name_embedding) <=> %s::vector "
            "LIMIT %s",
            (
                instruction_embedding,
                direction_id,
                instruction_embedding,
                self.TOP_K,
            ),
        )
        candidates: list[dict] = []
        for r in cur.fetchall():
            row = dict(r)
            if row.pop("similarity") < self.TERMS_SIM_THRESHOLD:
                continue
            candidates.append(row)

        if not candidates:
            return []

        id_to_cand = {c["id"]: c for c in candidates}
        candidate_lines = "\n".join(
            f"id={c['id']} | scope={c.get('scope', '')} | "
            f"name={c.get('name', '')} | "
            f"short_description={self._trim(c.get('short_description'))}"
            for c in candidates
        )
        context_parts = [self._format_direction_header(direction)]
        if matched_roles:
            context_parts.append(
                "# Выбранные роли\n"
                + "\n".join(f"- {r['name']}" for r in matched_roles)
            )
        context_parts.append("# Кандидаты-термины\n" + candidate_lines)

        task = (
            "Твоя задача — ВЫБРАТЬ из приведённого списка терминов те, которые\n"
            "нужны для понимания задачи аналитика или описания выбранных ролей.\n"
            "Ты НЕ отвечаешь на задачу, НЕ объясняешь термины своими словами.\n"
            "Ты только отбираешь записи из базы знаний.\n\n"
            "Термин подходит, если:\n"
            "- он называет понятие, встречающееся в задаче аналитика, ИЛИ\n"
            "- он описывает концепцию, которой оперирует выбранная роль в своей\n"
            "  работе и которая упоминается в формулировке задачи.\n\n"
            "Исключай термины, лишь косвенно относящиеся к теме. Если ни один\n"
            "не подходит — пустой список.\n\n"
            "НЕ добавляй id, отсутствующие в списке. НЕ выдумывай новые термины.\n"
            "Аббревиатуры расшифровывай ТОЛЬКО через список из «# Контекст».\n\n"
            "Финальный ответ СТРОГО формата:\n"
            '{"term_ids": [<int>, ...], "reasoning": "<1–2 предложения>"}\n'
            "Никаких комментариев после JSON."
        )

        steps = [
            "Перечитай задачу аналитика. В 1–2 предложениях выдели, какие понятия в ней встречаются.",
            "Разверни аббревиатуры из «# Контекст» (для себя, в рассуждениях).",
            "Пройди по КАЖДОМУ кандидату-термину отдельно. Дай одно предложение: «подходит, потому что ...» или «не подходит, потому что ...».",
            "Собери id подходящих. Если ни одного — пустой список.",
            "Напиши reasoning — 1–2 предложения про принцип отбора.",
            "Выведи финальный JSON строго по формату из «# Описания задачи».",
        ]

        prompt = self._build_prompt(
            task=task,
            title="Выборка терминов, связанных с задачей аналитика",
            text=f"Задача аналитика:\n{instruction}",
            steps=steps,
            context="\n\n".join(context_parts),
        )

        data = self._call_structured(
            prompt, validate=self._ids_validator("term_ids", id_to_cand)
        )
        return [
            id_to_cand[i]
            for i in dict.fromkeys(data["term_ids"])
            if i in id_to_cand
        ]

    def _select_metrics(
        self,
        cur,
        *,
        direction_id: int,
        matched_roles: list[dict],
        matched_role_ids: list[int],
        instruction: str,
        instruction_embedding: list[float],
        direction: dict,
    ) -> list[dict]:
        candidates = self._semantic_fetch(
            cur,
            table="rag_v2.metrics",
            direction_id=direction_id,
            embedding=instruction_embedding,
            role_ids=matched_role_ids,
            threshold=self.METRICS_SIM_THRESHOLD,
        )
        if not candidates:
            return []

        id_to_cand = {c["id"]: c for c in candidates}
        candidate_lines = "\n".join(
            f"id={c['id']} | name={c.get('name', '')} | "
            f"short_description={self._trim(c.get('short_description'))}"
            for c in candidates
        )
        context_parts = [self._format_direction_header(direction)]
        if matched_roles:
            context_parts.append(
                "# Выбранные роли\n"
                + "\n".join(f"- {r['name']}" for r in matched_roles)
            )
        context_parts.append("# Кандидаты-метрики\n" + candidate_lines)

        task = (
            "Твоя задача — ВЫБРАТЬ из приведённого списка метрик те, которые\n"
            "относятся к задаче аналитика. Ты НЕ отвечаешь на задачу, НЕ\n"
            "предлагаешь новых метрик, НЕ рассчитываешь ничего. Ты только\n"
            "отбираешь метрики, УЖЕ существующие в списке «# Контекст».\n\n"
            "Метрика подходит, если:\n"
            "- она измеряет работу выбранной роли, ИЛИ\n"
            "- она описывает показатель процесса, упомянутого в задаче, ИЛИ\n"
            "- её имя/описание прямо упоминает понятия из формулировки задачи.\n\n"
            "Отсекай метрики из смежных, но не запрошенных областей. Если ни\n"
            "одна не подходит — пустой список.\n\n"
            "НЕ добавляй id, отсутствующие в списке. НЕ выдумывай метрики.\n"
            "Аббревиатуры расшифровывай ТОЛЬКО через список из «# Контекст».\n\n"
            "Финальный ответ СТРОГО формата:\n"
            '{"metric_ids": [<int>, ...], "reasoning": "<1–2 предложения>"}\n'
            "Никаких комментариев после JSON."
        )

        steps = [
            "Перечитай задачу аналитика. В 1–2 предложениях выдели: про какой процесс и какой показатель.",
            "Разверни аббревиатуры из «# Контекст» (для себя, в рассуждениях).",
            "Пройди по КАЖДОЙ метрике-кандидату отдельно. Дай одно предложение: «подходит, потому что ...» или «не подходит, потому что ...».",
            "Собери id подходящих. Если ни одной — пустой список.",
            "Напиши reasoning — 1–2 предложения про принцип отбора.",
            "Выведи финальный JSON строго по формату из «# Описания задачи».",
        ]

        prompt = self._build_prompt(
            task=task,
            title="Выборка метрик из базы знаний",
            text=f"Задача аналитика:\n{instruction}",
            steps=steps,
            context="\n\n".join(context_parts),
        )

        data = self._call_structured(
            prompt, validate=self._ids_validator("metric_ids", id_to_cand)
        )
        return [
            id_to_cand[i]
            for i in dict.fromkeys(data["metric_ids"])
            if i in id_to_cand
        ]

    def _select_algorithms(
        self,
        cur,
        *,
        direction_id: int,
        matched_roles: list[dict],
        matched_role_ids: list[int],
        instruction: str,
        instruction_embedding: list[float],
        direction: dict,
    ) -> list[dict]:
        candidates = self._semantic_fetch(
            cur,
            table="rag_v2.algorithms",
            direction_id=direction_id,
            embedding=instruction_embedding,
            role_ids=matched_role_ids,
            threshold=self.ALGORITHMS_SIM_THRESHOLD,
        )
        if not candidates:
            return []

        id_to_cand = {c["id"]: c for c in candidates}
        candidate_lines = "\n".join(
            f"id={c['id']} | name={c.get('name', '')} | "
            f"short_description={self._trim(c.get('short_description'))}"
            for c in candidates
        )
        context_parts = [self._format_direction_header(direction)]
        if matched_roles:
            context_parts.append(
                "# Выбранные роли\n"
                + "\n".join(f"- {r['name']}" for r in matched_roles)
            )
        context_parts.append("# Кандидаты-алгоритмы\n" + candidate_lines)

        task = (
            "Твоя задача — ВЫБРАТЬ из приведённого списка алгоритмов и\n"
            "регламентов те, которые относятся к задаче аналитика. Ты НЕ\n"
            "выполняешь задачу, НЕ формулируешь новых алгоритмов, НЕ\n"
            "рассчитываешь по ним. Ты только отбираешь записи, УЖЕ\n"
            "присутствующие в списке «# Контекст».\n\n"
            "Алгоритм подходит, если:\n"
            "- он описывает процесс или регламент, про который спрашивает\n"
            "  задача, ИЛИ\n"
            "- он задаёт правила расчёта/интерпретации метрик из задачи, ИЛИ\n"
            "- он применяется выбранной ролью в её работе.\n\n"
            "Отсекай алгоритмы с лишь косвенным пересечением с темой. Если\n"
            "ни один не подходит — пустой список.\n\n"
            "НЕ добавляй id, отсутствующие в списке. НЕ выдумывай алгоритмы.\n"
            "Аббревиатуры расшифровывай ТОЛЬКО через список из «# Контекст».\n\n"
            "Финальный ответ СТРОГО формата:\n"
            '{"algorithm_ids": [<int>, ...], "reasoning": "<1–2 предложения>"}\n'
            "Никаких комментариев после JSON."
        )

        steps = [
            "Перечитай задачу аналитика. В 1–2 предложениях выдели: какой процесс/регламент/расчёт в ней описан.",
            "Разверни аббревиатуры из «# Контекст» (для себя, в рассуждениях).",
            "Пройди по КАЖДОМУ алгоритму-кандидату отдельно. Дай одно предложение: «подходит, потому что ...» или «не подходит, потому что ...».",
            "Собери id подходящих. Если ни одного — пустой список.",
            "Напиши reasoning — 1–2 предложения про принцип отбора.",
            "Выведи финальный JSON строго по формату из «# Описания задачи».",
        ]

        prompt = self._build_prompt(
            task=task,
            title="Выборка алгоритмов из базы знаний",
            text=f"Задача аналитика:\n{instruction}",
            steps=steps,
            context="\n\n".join(context_parts),
        )

        data = self._call_structured(
            prompt, validate=self._ids_validator("algorithm_ids", id_to_cand)
        )
        return [
            id_to_cand[i]
            for i in dict.fromkeys(data["algorithm_ids"])
            if i in id_to_cand
        ]

    # ==================================================================
    # Shared semantic fetch (metrics / algorithms)
    # ==================================================================

    def _semantic_fetch(
        self,
        cur,
        *,
        table: str,
        direction_id: int,
        embedding: list[float],
        role_ids: list[int],
        threshold: float,
    ) -> list[dict]:
        base_sql = (
            "SELECT id, name, short_description, detailed_description, "
            "       1 - (COALESCE(short_description_embedding, name_embedding) <=> %s::vector) AS similarity "
            f"FROM {table} "
            "WHERE direction_id = %s "
            "  AND (short_description_embedding IS NOT NULL "
            "       OR name_embedding IS NOT NULL) "
            "{role_clause}"
            " ORDER BY COALESCE(short_description_embedding, name_embedding) <=> %s::vector "
            "LIMIT %s"
        )

        def run(clause: str, extra_params: tuple) -> list[dict]:
            sql = base_sql.replace("{role_clause}", clause)
            params = (
                embedding,
                direction_id,
                *extra_params,
                embedding,
                self.TOP_K,
            )
            cur.execute(sql, params)
            rows: list[dict] = []
            for r in cur.fetchall():
                row = dict(r)
                if row.pop("similarity") < threshold:
                    continue
                rows.append(row)
            return rows

        if role_ids:
            rows = run(" AND role_ids && %s::bigint[]", (role_ids,))
            if rows:
                return rows
        return run("", ())

    # ==================================================================
    # Validators
    # ==================================================================

    @staticmethod
    def _ids_validator(
        id_key: str, id_to_cand: dict
    ) -> Callable[[Any], None]:
        valid = set(id_to_cand.keys())

        def validate(data: Any) -> None:
            if not isinstance(data, dict):
                raise ValueError("expected JSON object")
            ids = data.get(id_key)
            if not isinstance(ids, list) or not all(
                isinstance(i, int) for i in ids
            ):
                raise ValueError(f"{id_key} must be a list of integers")
            for i in ids:
                if i not in valid:
                    raise ValueError(
                        f"id {i} in {id_key} is not among the candidates"
                    )
            reasoning = data.get("reasoning")
            if not isinstance(reasoning, str) or not reasoning.strip():
                raise ValueError("reasoning must be a non-empty string")

        return validate


def _extract_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*([\[{].*?[\]}])\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    dec = json.JSONDecoder()
    last_obj: Any = None
    i = 0
    n = len(text)
    while i < n:
        if text[i] in "{[":
            try:
                obj, end = dec.raw_decode(text[i:])
                last_obj = obj
                i += end
                continue
            except json.JSONDecodeError:
                pass
        i += 1
    if last_obj is not None:
        return last_obj
    raise json.JSONDecodeError("no JSON in LLM output", text, 0)
