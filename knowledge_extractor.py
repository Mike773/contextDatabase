"""Подготовка бизнес-контекста для агента-аналитика из базы знаний rag_v2.

Класс KnowledgeExtractor принимает direction_id + роль + сценарий +
инструкцию + формат ответа, после чего отдаёт пять блоков контекста:
информация об организации, о направлении, о ситуации (роль × сценарий ×
инструкция), детальное описание релевантных алгоритмов (с обращением к
документам-первоисточникам) и текстовый список имён метрик, связанных
с этими алгоритмами.

Файл сознательно self-contained: ничего не импортируется из соседних
модулей проекта — только внешние библиотеки. Его можно скопировать в
любой сторонний скрипт одним файлом. Логика отбора (embedding + LLM
verification с пошаговым reasoning) и ключевые утилиты продублированы
из extractor.py по этой причине.
"""

import json
import re
from typing import Any, Callable

try:
    import psycopg2 as _psycopg
    import psycopg2.extras as _psycopg_extras
    from pgvector.psycopg2 import register_vector as _register_vector

    _PSYCOPG_VERSION = 2
except ImportError:  # pragma: no cover
    try:
        import psycopg as _psycopg
        from psycopg.rows import dict_row as _psycopg_dict_row
        from pgvector.psycopg import register_vector as _register_vector

        _PSYCOPG_VERSION = 3
    except ImportError as _e:  # pragma: no cover
        raise ImportError(
            "knowledge_extractor requires either psycopg2 "
            "(with pgvector.psycopg2) or psycopg v3 "
            "(with pgvector.psycopg)."
        ) from _e


def _pg_connect(dsn: str):
    conn = _psycopg.connect(dsn)
    conn.autocommit = True
    _register_vector(conn)
    return conn


def _pg_dict_cursor(conn):
    if _PSYCOPG_VERSION == 2:
        return conn.cursor(cursor_factory=_psycopg_extras.RealDictCursor)
    return conn.cursor(row_factory=_psycopg_dict_row)


class KnowledgeExtractor:
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
        self._ready = False

        self._direction_id: int | None = None
        self._role_query: str | None = None
        self._scenario: str | None = None
        self._instruction: str | None = None
        self._response_format: str | None = None

        self._direction: dict | None = None
        self._abbreviations: dict = {}
        self._org_terms: list[dict] = []
        self._org_claims: list[dict] = []
        self._dir_terms: list[dict] = []
        self._dir_claims: list[dict] = []
        self._roles: list[dict] = []
        self._matched_roles: list[dict] = []
        self._scenario_hits: dict = {"claims": [], "terms": []}
        self._instruction_hits: dict = {"claims": [], "terms": []}
        self._situational_summary: str | None = None
        self._metrics: list[dict] = []
        self._algorithms: list[dict] = []
        self._algorithms_text: str | None = None
        self._organization_text: str | None = None
        self._direction_text: str | None = None

    # ==================================================================
    # Public API
    # ==================================================================

    def build_context(
        self,
        *,
        direction_id: int,
        role_query: str,
        scenario: str,
        instruction: str,
        response_format: str,
    ) -> None:
        self._direction_id = direction_id
        self._role_query = role_query
        self._scenario = scenario
        self._instruction = instruction
        self._response_format = response_format

        conn = _pg_connect(self._dsn)
        try:
            with _pg_dict_cursor(conn) as cur:
                direction = self._fetch_direction(cur, direction_id)
                if direction is None:
                    raise ValueError(f"direction {direction_id} not found")
                self._direction = direction
                self._abbreviations = direction.get("abbreviations") or {}

                self._roles = self._fetch_roles(cur, direction_id)
                self._org_terms = self._fetch_terms(
                    cur, direction_id, "organization"
                )
                self._org_claims = self._fetch_claims_full(
                    cur, direction_id, "organization"
                )
                self._dir_terms = self._fetch_terms(
                    cur, direction_id, "direction"
                )
                self._dir_claims = self._fetch_claims_full(
                    cur, direction_id, "direction"
                )

                combined_query = self._format_combined_query(
                    role_query=role_query,
                    scenario=scenario,
                    instruction=instruction,
                )
                scenario_embedding = self._embed_for_retrieval(
                    scenario, self._abbreviations
                )
                instruction_embedding = self._embed_for_retrieval(
                    instruction, self._abbreviations
                )

                self._matched_roles = self._match_roles(
                    self._roles, combined_query, direction
                )
                matched_role_ids = [r["id"] for r in self._matched_roles]

                self._scenario_hits = self._select_scenario_hits(
                    cur,
                    direction_id=direction_id,
                    scenario=scenario,
                    scenario_embedding=scenario_embedding,
                    matched_roles=self._matched_roles,
                    direction=direction,
                )
                self._instruction_hits = self._select_instruction_hits(
                    matched_roles=self._matched_roles,
                    scenario_hits=self._scenario_hits,
                    instruction=instruction,
                    direction=direction,
                )
                self._situational_summary = self._summarize_situational_context(
                    matched_roles=self._matched_roles,
                    scenario_hits=self._scenario_hits,
                    instruction_hits=self._instruction_hits,
                    role_query=role_query,
                    scenario=scenario,
                    instruction=instruction,
                    direction=direction,
                )

                self._algorithms = self._select_algorithms(
                    cur,
                    direction_id=direction_id,
                    matched_roles=self._matched_roles,
                    matched_role_ids=matched_role_ids,
                    instruction=instruction,
                    instruction_embedding=instruction_embedding,
                    direction=direction,
                )

                algorithm_metric_ids = sorted(
                    {
                        mid
                        for a in self._algorithms
                        for mid in (a.get("metric_ids") or [])
                    }
                )
                self._metrics = (
                    self._fetch_metrics_by_ids(
                        cur, direction_id, algorithm_metric_ids
                    )
                    if algorithm_metric_ids
                    else []
                )

                if self._algorithms:
                    algorithm_document_ids = sorted(
                        {
                            did
                            for a in self._algorithms
                            for did in (a.get("document_ids") or [])
                        }
                    )
                    source_documents = (
                        self._fetch_documents_by_ids(
                            cur, direction_id, algorithm_document_ids
                        )
                        if algorithm_document_ids
                        else []
                    )
                    self._algorithms_text = self._describe_algorithms(
                        algorithms=self._algorithms,
                        algorithm_metrics=self._metrics,
                        source_documents=source_documents,
                        instruction=instruction,
                        response_format=response_format,
                        direction=direction,
                    )
                else:
                    self._algorithms_text = None

                self._organization_text = self._build_organization_text(
                    org_terms=self._org_terms,
                    org_claims=self._org_claims,
                    direction=direction,
                )
                self._direction_text = self._compose_direction_body(
                    dir_terms=self._dir_terms,
                    dir_claims=self._dir_claims,
                    direction=direction,
                )
        finally:
            conn.close()

        self._ready = True

    def organization_info(self) -> str:
        self._require_ready()
        if not self._org_terms and not self._org_claims:
            return self.NOT_FOUND
        return self._organization_text or self.NOT_FOUND

    def direction_info(self) -> str:
        self._require_ready()
        return self._direction_text or self.NOT_FOUND

    def situational_context(self) -> str:
        self._require_ready()
        return self._situational_summary or self.NOT_FOUND

    def related_algorithms(self) -> str:
        self._require_ready()
        return self._algorithms_text or self.NOT_FOUND

    def related_metrics(self) -> str:
        self._require_ready()
        if not self._metrics:
            return self.NOT_FOUND
        return "\n".join(
            f"- {m.get('name', '')}"
            for m in self._metrics
            if m.get("name")
        ) or self.NOT_FOUND

    def _require_ready(self) -> None:
        if not self._ready:
            raise RuntimeError(
                "build_context must be called before reading context blocks"
            )

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
            "SELECT id, name, short_description, detailed_description, "
            "       alternative_names "
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

    BACKREF_PATTERNS = (
        r"см\.\s*выше",
        r"как\s+(?:приведено|указано|описано|сказано)\s+(?:выше|ранее)",
        r"приведено\s+выше",
        r"описано\s+выше",
        r"описание\s+выше",
        r"смотр(?:и|ите)\s+выше",
        r"в\s+шаг(?:ах|е)\s+выше",
        r"см\.\s*пункт",
    )

    def _call_free_text(self, prompt: str) -> str:
        last_err: Exception | None = None
        current_prompt = prompt
        for attempt in range(self.MAX_LLM_RETRIES):
            response = self._llm_call(current_prompt)
            text = _extract_after_marker(response, "ИТОГ:")
            if text and not self._has_backreference(text):
                return text
            if not text:
                last_err = ValueError("marker 'ИТОГ:' not found in response")
                current_prompt = (
                    prompt
                    + f"\n\n# Предыдущая попытка {attempt + 1} не выдала маркер «ИТОГ:».\n"
                    "Повтори выполнение по шагам и ОБЯЗАТЕЛЬНО завершающей строкой"
                    " выведи «ИТОГ:» и после него — финальный текст одним абзацем."
                )
            else:
                last_err = ValueError(
                    "финальный ответ ссылается на рассуждения выше"
                )
                current_prompt = (
                    prompt
                    + f"\n\n# Предыдущая попытка {attempt + 1} выдала «ИТОГ:», "
                    "но внутри финального ответа есть ссылки на рассуждения "
                    "выше («см. выше», «как приведено ранее», «описание выше» "
                    "и т.п.). Читатель их не видит.\n"
                    "Повтори выполнение: сохрани рассуждения по шагам, а "
                    "ПОСЛЕ «ИТОГ:» ПОЛНОСТЬЮ воспроизведи весь нужный текст "
                    "без ссылок на ранее написанное — целостным, "
                    "самодостаточным блоком."
                )
        raise RuntimeError(
            f"free-text LLM call failed after {self.MAX_LLM_RETRIES} attempts: {last_err}"
        )

    @classmethod
    def _has_backreference(cls, text: str) -> bool:
        lowered = text.lower()
        return any(
            re.search(p, lowered) for p in cls.BACKREF_PATTERNS
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
            "Финальный ответ ДОЛЖЕН БЫТЬ САМОДОСТАТОЧНЫМ: читатель увидит "
            "ТОЛЬКО его и НЕ увидит рассуждений по шагам. ЗАПРЕЩЕНЫ ссылки "
            "«как приведено выше», «см. выше», «описание в шагах выше», "
            "«см. пункт N» и любые отсылки к ранее написанному тексту. Всё "
            "содержимое, необходимое читателю, должно быть полностью "
            "воспроизведено внутри финального ответа.\n\n"
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

    @staticmethod
    def _format_combined_query(
        *, role_query: str, scenario: str, instruction: str
    ) -> str:
        return (
            f"Должность/роль, о которой спрашивают: {role_query}\n"
            f"Сценарий (процесс, в рамках которого идёт обращение): {scenario}\n"
            f"Инструкция (вопрос, на который надо ответить): {instruction}"
        )

    # ==================================================================
    # Role matching
    # ==================================================================

    def _match_roles(
        self, roles: list[dict], instruction: str, direction: dict
    ) -> list[dict]:
        if not roles:
            return []

        role_lines = "\n".join(
            f"id={r['id']} | name={r['name']} | "
            f"alternative_names={list(r.get('alternative_names') or [])} | "
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
            "  побуквенное совпадение. Учитывай столбец alternative_names.\n"
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
            "Выпиши список всех ролей из «# Контекст» — id, name и alternative_names.",
            "Прочитай задачу аналитика в «# Документ». В 1–2 предложениях опиши, о какой должности/роли/функции идёт речь.",
            "Разверни в задаче аббревиатуры из «# Контекст» (для себя, в рассуждениях). Незнакомые не расшифровывай.",
            "Для КАЖДОЙ роли из списка отдельно дай вердикт — одно предложение «подходит, потому что ...» или «не подходит, потому что ...». При проверке учитывай и каноническое name, и alternative_names.",
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
    # Scenario / instruction selectors (situational context)
    # ==================================================================

    def _select_scenario_hits(
        self,
        cur,
        *,
        direction_id: int,
        scenario: str,
        scenario_embedding: list[float],
        matched_roles: list[dict],
        direction: dict,
    ) -> dict:
        claim_candidates = self._fetch_claim_candidates(
            cur, direction_id, scenario_embedding, self.CLAIMS_SIM_THRESHOLD
        )
        term_candidates = self._fetch_term_candidates(
            cur, direction_id, scenario_embedding, self.TERMS_SIM_THRESHOLD
        )

        if not claim_candidates and not term_candidates:
            return {"claims": [], "terms": []}

        claim_index = {c["id"]: c for c in claim_candidates}
        term_index = {t["id"]: t for t in term_candidates}

        claim_lines = "\n".join(
            f"id={c['id']} | scope={c.get('scope', '')} | "
            f"role_names={list(c.get('role_names') or [])} | "
            f"short={self._trim(c.get('short_description'))} | "
            f"detailed={self._trim(c.get('detailed_description'))}"
            for c in claim_candidates
        ) or "— нет кандидатов —"
        term_lines = "\n".join(
            f"id={t['id']} | scope={t.get('scope', '')} | "
            f"name={t.get('name', '')} | "
            f"short={self._trim(t.get('short_description'))} | "
            f"detailed={self._trim(t.get('detailed_description'))}"
            for t in term_candidates
        ) or "— нет кандидатов —"

        context_parts = [self._format_direction_header(direction)]
        if matched_roles:
            context_parts.append(
                "# Выбранные роли\n"
                + "\n".join(
                    f"- {r['name']}: {self._trim(r.get('short_description'))}"
                    for r in matched_roles
                )
            )
        else:
            context_parts.append(
                "# Выбранные роли\nРоли не определены: по данной роли "
                "информации нет."
            )
        context_parts.append("# Кандидаты-утверждения\n" + claim_lines)
        context_parts.append("# Кандидаты-термины\n" + term_lines)

        task = (
            "Твоя задача — ВЫБРАТЬ из списка кандидатов утверждения и термины,\n"
            "которые описывают указанный бизнес-процесс (сценарий) или\n"
            "относятся к работе выбранных ролей в рамках этого процесса. Ты НЕ\n"
            "объясняешь сценарий своими словами, НЕ выдумываешь новых фактов,\n"
            "только отбираешь записи из списка.\n\n"
            "Кандидат подходит, если:\n"
            "- он прямо описывает шаги/правила/ограничения данного процесса, ИЛИ\n"
            "- он называет/определяет понятие, без которого процесс не\n"
            "  понимается, ИЛИ\n"
            "- он фиксирует вклад одной из выбранных ролей в этот процесс.\n\n"
            "Отсекай кандидатов, лишь косвенно связанных со сценарием. Если ни\n"
            "один не подходит — пустые списки.\n\n"
            "НЕ добавляй id, отсутствующие в списках. Аббревиатуры расшифровывай\n"
            "ТОЛЬКО через список из «# Контекст».\n\n"
            "Финальный ответ СТРОГО формата:\n"
            '{"claim_ids": [<int>, ...], "term_ids": [<int>, ...], '
            '"reasoning": "<1–2 предложения>"}\n'
            "Никаких комментариев после JSON."
        )

        steps = [
            "Перечитай сценарий в «# Документ». В 1–2 предложениях выдели ключевые шаги процесса и участников.",
            "Разверни аббревиатуры из «# Контекст» (для себя, в рассуждениях).",
            "Пройди по КАЖДОМУ утверждению-кандидату отдельно. Дай одно предложение: «подходит, потому что ...» или «не подходит, потому что ...».",
            "Пройди по КАЖДОМУ термину-кандидату отдельно по тому же принципу.",
            "Собери id подходящих утверждений и терминов раздельно. Если пусто — пустой список.",
            "Напиши reasoning — 1–2 предложения про принцип отбора.",
            "Выведи финальный JSON строго по формату из «# Описания задачи».",
        ]

        prompt = self._build_prompt(
            task=task,
            title="Отбор утверждений и терминов под сценарий",
            text=f"Сценарий (процесс):\n{scenario}",
            steps=steps,
            context="\n\n".join(context_parts),
        )

        data = self._call_structured(
            prompt,
            validate=self._pair_ids_validator(
                claim_ids_key="claim_ids",
                term_ids_key="term_ids",
                claim_index=claim_index,
                term_index=term_index,
            ),
        )
        return {
            "claims": [
                claim_index[i]
                for i in dict.fromkeys(data["claim_ids"])
                if i in claim_index
            ],
            "terms": [
                term_index[i]
                for i in dict.fromkeys(data["term_ids"])
                if i in term_index
            ],
        }

    def _select_instruction_hits(
        self,
        *,
        matched_roles: list[dict],
        scenario_hits: dict,
        instruction: str,
        direction: dict,
    ) -> dict:
        scenario_claims = scenario_hits.get("claims") or []
        scenario_terms = scenario_hits.get("terms") or []
        if not scenario_claims and not scenario_terms:
            return {"claims": [], "terms": []}

        claim_index = {c["id"]: c for c in scenario_claims}
        term_index = {t["id"]: t for t in scenario_terms}

        claim_lines = "\n".join(
            f"id={c['id']} | scope={c.get('scope', '')} | "
            f"short={self._trim(c.get('short_description'))} | "
            f"detailed={self._trim(c.get('detailed_description'))}"
            for c in scenario_claims
        ) or "— нет кандидатов —"
        term_lines = "\n".join(
            f"id={t['id']} | scope={t.get('scope', '')} | "
            f"name={t.get('name', '')} | "
            f"short={self._trim(t.get('short_description'))} | "
            f"detailed={self._trim(t.get('detailed_description'))}"
            for t in scenario_terms
        ) or "— нет кандидатов —"

        context_parts = [self._format_direction_header(direction)]
        if matched_roles:
            context_parts.append(
                "# Выбранные роли\n"
                + "\n".join(
                    f"- {r['name']}: {self._trim(r.get('short_description'))}"
                    for r in matched_roles
                )
            )
        else:
            context_parts.append(
                "# Выбранные роли\nРоли не определены: по данной роли "
                "информации нет."
            )
        context_parts.append(
            "# Утверждения, отобранные под сценарий\n" + claim_lines
        )
        context_parts.append(
            "# Термины, отобранные под сценарий\n" + term_lines
        )

        task = (
            "Твоя задача — ИЗ уже отобранных под сценарий утверждений и\n"
            "терминов оставить ТОЛЬКО те, что напрямую нужны, чтобы ответить\n"
            "на инструкцию. Это СУЖЕНИЕ списка — нельзя добавлять новые id,\n"
            "можно только исключать.\n\n"
            "Кандидат подходит, если:\n"
            "- он содержит факт/правило, которое используется для ответа на\n"
            "  инструкцию, ИЛИ\n"
            "- без него ответ на инструкцию был бы неточным или неполным.\n\n"
            "Если кандидат нужен только для понимания сценария в целом, но не\n"
            "для конкретной инструкции — исключай. Если подходящих нет — верни\n"
            "пустые списки.\n\n"
            "НЕ добавляй id, отсутствующие в «# Контекст». Аббревиатуры\n"
            "расшифровывай ТОЛЬКО через список из «# Контекст».\n\n"
            "Финальный ответ СТРОГО формата:\n"
            '{"claim_ids": [<int>, ...], "term_ids": [<int>, ...], '
            '"reasoning": "<1–2 предложения>"}\n'
            "Никаких комментариев после JSON."
        )

        steps = [
            "Перечитай инструкцию в «# Документ». В 1–2 предложениях зафиксируй, что именно нужно ответить.",
            "Разверни аббревиатуры из «# Контекст» (для себя, в рассуждениях).",
            "Пройди по КАЖДОМУ утверждению из списка отобранных под сценарий. Дай одно предложение: «нужно для инструкции, потому что ...» или «не нужно, потому что ...».",
            "Пройди по КАЖДОМУ термину по тому же принципу.",
            "Собери id нужных утверждений и терминов раздельно. Если пусто — пустой список.",
            "Напиши reasoning — 1–2 предложения про принцип отбора.",
            "Выведи финальный JSON строго по формату из «# Описания задачи».",
        ]

        prompt = self._build_prompt(
            task=task,
            title="Сужение списка фактов под инструкцию",
            text=f"Инструкция:\n{instruction}",
            steps=steps,
            context="\n\n".join(context_parts),
        )

        data = self._call_structured(
            prompt,
            validate=self._pair_ids_validator(
                claim_ids_key="claim_ids",
                term_ids_key="term_ids",
                claim_index=claim_index,
                term_index=term_index,
            ),
        )
        return {
            "claims": [
                claim_index[i]
                for i in dict.fromkeys(data["claim_ids"])
                if i in claim_index
            ],
            "terms": [
                term_index[i]
                for i in dict.fromkeys(data["term_ids"])
                if i in term_index
            ],
        }

    def _summarize_situational_context(
        self,
        *,
        matched_roles: list[dict],
        scenario_hits: dict,
        instruction_hits: dict,
        role_query: str,
        scenario: str,
        instruction: str,
        direction: dict,
    ) -> str:
        roles_block = (
            "\n".join(
                f"- {r['name']}: {self._trim(r.get('short_description'))} "
                f"(альт. имена: {list(r.get('alternative_names') or [])})"
                for r in matched_roles
            )
            if matched_roles
            else "НЕ НАЙДЕНО: по данной роли информации нет."
        )

        scenario_claims = scenario_hits.get("claims") or []
        scenario_terms = scenario_hits.get("terms") or []
        if scenario_claims or scenario_terms:
            scenario_block_parts = []
            if scenario_claims:
                scenario_block_parts.append(
                    "Утверждения:\n"
                    + "\n".join(
                        f"- {self._trim(c.get('short_description'))}: "
                        f"{self._trim(c.get('detailed_description'), 600)}"
                        for c in scenario_claims
                    )
                )
            if scenario_terms:
                scenario_block_parts.append(
                    "Термины:\n"
                    + "\n".join(
                        f"- {t.get('name', '')}: "
                        f"{self._trim(t.get('short_description'))}"
                        for t in scenario_terms
                    )
                )
            scenario_block = "\n\n".join(scenario_block_parts)
        else:
            scenario_block = "НЕ НАЙДЕНО: информации по сценарию не нашёл."

        instruction_claims = instruction_hits.get("claims") or []
        instruction_terms = instruction_hits.get("terms") or []
        if instruction_claims or instruction_terms:
            instruction_block_parts = []
            if instruction_claims:
                instruction_block_parts.append(
                    "Утверждения:\n"
                    + "\n".join(
                        f"- {self._trim(c.get('short_description'))}: "
                        f"{self._trim(c.get('detailed_description'), 600)}"
                        for c in instruction_claims
                    )
                )
            if instruction_terms:
                instruction_block_parts.append(
                    "Термины:\n"
                    + "\n".join(
                        f"- {t.get('name', '')}: "
                        f"{self._trim(t.get('short_description'))}"
                        for t in instruction_terms
                    )
                )
            instruction_block = "\n\n".join(instruction_block_parts)
        else:
            instruction_block = "НЕ НАЙДЕНО: информации по инструкции не нашёл."

        context = (
            f"{self._format_direction_header(direction)}\n\n"
            "# Входные параметры\n"
            f"Роль/должность, про которую спрашивают: {role_query}\n"
            f"Сценарий: {scenario}\n"
            f"Инструкция: {instruction}\n\n"
            f"# Выбранные роли\n{roles_block}\n\n"
            f"# Знания по сценарию\n{scenario_block}\n\n"
            f"# Знания по инструкции\n{instruction_block}"
        )

        task = (
            "Твоя задача — написать СВЯЗНЫЙ абзац на русском (4–7 предложений),\n"
            "описывающий контекст ситуации: про какие роли идёт речь, в рамках\n"
            "какого процесса, и какие факты из базы знаний применимы к\n"
            "инструкции. Используй ТОЛЬКО факты из блока «# Контекст».\n\n"
            "Правила:\n"
            "- Если блок помечен «НЕ НАЙДЕНО» — явно скажи, что информации\n"
            "  нет, не придумывай её.\n"
            "- Не переписывай задание и инструкцию дословно, формулируй\n"
            "  по-своему, опираясь на факты.\n"
            "- Аббревиатуры в итоговом тексте разворачивай через список из\n"
            "  «# Контекст». Незнакомые — не расшифровывай.\n"
            "- НЕ давай рекомендаций, НЕ решай инструкцию. Только контекст.\n\n"
            "Финальный ответ: после последнего шага выведи отдельной строкой\n"
            "«ИТОГ:», а следом одним абзацем — контекст ситуации."
        )

        steps = [
            "Прочитай входные параметры и все блоки фактов.",
            "Определи, кого описывают выбранные роли (или зафиксируй, что не нашёл).",
            "Сформулируй в 1–2 предложениях суть процесса (сценария) по найденным фактам или зафиксируй «НЕ НАЙДЕНО».",
            "Сформулируй в 1–2 предложениях, какие факты из базы знаний применимы к инструкции или зафиксируй «НЕ НАЙДЕНО».",
            "Собери всё в связный абзац, расшифровывая аббревиатуры по списку.",
            "После последнего шага выведи «ИТОГ:» на новой строке и сразу за ним — финальный абзац.",
        ]

        prompt = self._build_prompt(
            task=task,
            title="Описание контекста ситуации для агента-аналитика",
            text="См. входные параметры в «# Контекст».",
            steps=steps,
            context=context,
        )
        return self._call_free_text(prompt)

    # ==================================================================
    # Organization / direction text
    # ==================================================================

    def _build_organization_text(
        self,
        *,
        org_terms: list[dict],
        org_claims: list[dict],
        direction: dict,
    ) -> str | None:
        if not org_terms and not org_claims:
            return None

        claim_lines = (
            "\n".join(
                f"- {self._trim(c.get('short_description'))}: "
                f"{self._trim(c.get('detailed_description'), 800)}"
                for c in org_claims
            )
            or "— утверждений scope='organization' нет —"
        )
        term_lines = (
            "\n".join(
                f"- {t.get('name', '')}: "
                f"{self._trim(t.get('short_description'))} | "
                f"{self._trim(t.get('detailed_description'), 600)}"
                for t in org_terms
            )
            or "— терминов scope='organization' нет —"
        )

        context = (
            f"{self._format_direction_header(direction)}\n\n"
            f"# Утверждения об организации\n{claim_lines}\n\n"
            f"# Термины об организации\n{term_lines}"
        )

        task = (
            "Твоя задача — собрать СТРОКУ строго в формате:\n"
            "«Ты аналитик в компании <Название компании>, которая является "
            "<описание компании по фактам>».\n\n"
            "Правила:\n"
            "- Название компании возьми ТОЛЬКО из переданных фактов. Если оно\n"
            "  явно нигде не упомянуто — подставь «нашей компании» (именно в\n"
            "  такой форме, со строчной буквы).\n"
            "- Описание — связный текст, охватывающий ВСЕ переданные факты.\n"
            "  Не теряй детали (цифры, ограничения, названия подразделений),\n"
            "  но и не повторяй одно и то же.\n"
            "- НЕ добавляй выводов, НЕ придумывай фактов, НЕ ссылайся на то,\n"
            "  чего нет в «# Контекст».\n"
            "- Аббревиатуры в итоговом тексте разворачивай через список из\n"
            "  «# Контекст». Незнакомые — не расшифровывай.\n\n"
            "Финальный ответ: после последнего шага выведи отдельной строкой\n"
            "«ИТОГ:», а следом одной строкой (без переносов) — итоговый текст\n"
            "в формате выше."
        )

        steps = [
            "Выпиши ключевые факты из «# Утверждения об организации» и «# Термины об организации».",
            "Найди в фактах явное упоминание названия компании. Если не нашёл — пометь, что его нет.",
            "Сгруппируй факты по смыслу (чем занимается, где работает, какие подразделения/направления, какие ограничения).",
            "Собери описание компании одним предложением или списком обстоятельств, не теряя деталей.",
            "Склей итоговую строку по шаблону «Ты аналитик в компании <...>, которая является <...>».",
            "После последнего шага выведи «ИТОГ:» на новой строке и сразу за ним — итоговую строку.",
        ]

        prompt = self._build_prompt(
            task=task,
            title="Формирование строки про организацию",
            text="См. факты в «# Контекст».",
            steps=steps,
            context=context,
        )
        return self._call_free_text(prompt)

    def _compose_direction_body(
        self,
        *,
        dir_terms: list[dict],
        dir_claims: list[dict],
        direction: dict,
    ) -> str:
        parts: list[str] = []

        general_info = (direction.get("general_info") or "").strip()
        if general_info:
            parts.append(general_info)

        if dir_terms:
            terms_block = "\n".join(
                f"- {t.get('name', '')}: "
                f"{self._trim(t.get('short_description'))}"
                + (
                    f". {self._trim(t.get('detailed_description'), 600)}"
                    if t.get("detailed_description")
                    else ""
                )
                for t in dir_terms
            )
            parts.append(f"Ключевые термины направления:\n{terms_block}")

        if dir_claims:
            claims_block = "\n".join(
                f"- {self._trim(c.get('short_description'))}"
                + (
                    f". {self._trim(c.get('detailed_description'), 600)}"
                    if c.get("detailed_description")
                    else ""
                )
                for c in dir_claims
            )
            parts.append(f"Ключевые факты о направлении:\n{claims_block}")

        full_name = direction.get("full_name") or direction.get("short_name") or "направления"

        if not parts:
            return self.NOT_FOUND

        body = "\n\n".join(parts)
        return (
            f"Твоя задача помогать отвечать на вопросы подразделения "
            f"{full_name}, которое представляет собой:\n\n{body}"
        )

    # ==================================================================
    # Algorithms selector + metrics / documents lookup
    # ==================================================================

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
            extra_columns="document_ids, metric_ids, role_ids",
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

    @staticmethod
    def _fetch_metrics_by_ids(
        cur, direction_id: int, metric_ids: list[int]
    ) -> list[dict]:
        if not metric_ids:
            return []
        cur.execute(
            "SELECT id, name, short_description, detailed_description, "
            "       connections_description, role_ids "
            "FROM rag_v2.metrics "
            "WHERE direction_id = %s AND id = ANY(%s::bigint[]) "
            "ORDER BY id",
            (direction_id, metric_ids),
        )
        return [dict(r) for r in cur.fetchall()]

    @staticmethod
    def _fetch_documents_by_ids(
        cur, direction_id: int, document_ids: list[int]
    ) -> list[dict]:
        if not document_ids:
            return []
        cur.execute(
            "SELECT id, title, text "
            "FROM rag_v2.documents "
            "WHERE direction_id = %s AND id = ANY(%s::bigint[]) "
            "ORDER BY id",
            (direction_id, document_ids),
        )
        return [dict(r) for r in cur.fetchall()]

    def _describe_algorithms(
        self,
        *,
        algorithms: list[dict],
        algorithm_metrics: list[dict],
        source_documents: list[dict],
        instruction: str,
        response_format: str,
        direction: dict,
    ) -> str:
        if not algorithms:
            return self.NOT_FOUND

        metrics_by_id = {m["id"]: m for m in algorithm_metrics}

        algorithm_blocks: list[str] = []
        for a in algorithms:
            metric_parts = []
            for mid in a.get("metric_ids") or []:
                m = metrics_by_id.get(mid)
                if not m:
                    continue
                metric_parts.append(
                    f"  - {m.get('name', '')}: "
                    f"{m.get('short_description', '') or ''}\n"
                    f"    Подробно: {m.get('detailed_description', '') or ''}\n"
                    f"    Связи: {m.get('connections_description', '') or ''}"
                )
            metric_block = (
                "\n".join(metric_parts)
                if metric_parts
                else "  — связанные метрики не найдены —"
            )
            algorithm_blocks.append(
                f"Алгоритм id={a['id']} «{a.get('name', '')}»\n"
                f"Краткое описание: {a.get('short_description', '') or ''}\n"
                f"Полное описание из базы знаний:\n"
                f"{a.get('detailed_description', '') or ''}\n"
                f"Связанные метрики:\n{metric_block}\n"
                f"document_ids: {list(a.get('document_ids') or [])}"
            )
        algorithms_context = "\n\n".join(algorithm_blocks)

        if source_documents:
            doc_parts = []
            for d in source_documents:
                doc_parts.append(
                    f"Документ id={d['id']} «{d.get('title', '')}»:\n"
                    f"{d.get('text', '') or ''}"
                )
            documents_context = "\n\n".join(doc_parts)
        else:
            documents_context = (
                "— документы-первоисточники к этим алгоритмам в базе "
                "отсутствуют или не связаны —"
            )

        context = (
            f"{self._format_direction_header(direction)}\n\n"
            f"# Формат итогового ответа агента-аналитика\n"
            f"{response_format}\n\n"
            f"# Отобранные алгоритмы\n{algorithms_context}\n\n"
            f"# Документы-первоисточники\n{documents_context}"
        )

        task = (
            "Твоя задача — по каждому алгоритму из «# Отобранные алгоритмы»\n"
            "собрать МАКСИМАЛЬНО ДЕТАЛЬНОЕ текстовое описание, которое\n"
            "аналитик будет использовать напрямую для применения алгоритма и\n"
            "интерпретации его метрик. Это НЕ саммари и НЕ краткий пересказ.\n\n"
            "Агент-аналитик ответит пользователю в формате, указанном в\n"
            "«# Формат итогового ответа агента-аналитика». Используй этот\n"
            "формат как ПОДСКАЗКУ, какие сущности обязательно нужно вытащить\n"
            "из документов-первоисточников: если формат подразумевает цифры,\n"
            "перечисли все встречающиеся пороги и целевые значения; если\n"
            "таблицу — перечисли все столбцы/измерения из документа; и т.д.\n\n"
            "Правила:\n"
            "- КРИТИЧНО: если в документе-первоисточнике встречаются ИМЕНА\n"
            "  КОНКРЕТНЫХ МЕТРИК, ПОКАЗАТЕЛЕЙ, KPI или нормативных параметров —\n"
            "  ты ОБЯЗАН перечислить их ВСЕ в описании алгоритма, даже если\n"
            "  они НЕ привязаны к алгоритму в базе знаний. Каждую такую\n"
            "  метрику указывай по её точному имени из документа, рядом —\n"
            "  контекст (где упомянута, какое значение, какая формула).\n"
            "- Сохрани ВСЕ шаги, формулы, пороги, условия, исключения,\n"
            "  обозначенные как в базе знаний, так и в документах-\n"
            "  первоисточниках. Если документ-первоисточник добавляет\n"
            "  уточнения к шагам — включи их в описание.\n"
            "- НЕ сокращай числа, сроки, ссылки на системы и роли.\n"
            "- Обязательно перечисли связанные метрики с их целевыми\n"
            "  значениями и правилами интерпретации.\n"
            "- Если для какого-то алгоритма документов-первоисточников нет —\n"
            "  работай только с его полным описанием из базы; не придумывай\n"
            "  деталей, которых нет в «# Контекст».\n"
            "- Аббревиатуры разворачивай ТОЛЬКО через список из «# Контекст».\n"
            "- НЕ отвечай на саму инструкцию — готовь справочный материал.\n\n"
            "Финальный ответ: после последнего шага выведи отдельной строкой\n"
            "«ИТОГ:», а следом — детальное описание каждого алгоритма одним\n"
            "текстом. Для каждого начинай с заголовка «Алгоритм: <имя>». В\n"
            "конце описания каждого алгоритма добавь подзаголовок «Метрики и\n"
            "показатели, упомянутые в документах» — маркированный список\n"
            "имён всех метрик/показателей/KPI, встреченных в документах-\n"
            "первоисточниках этого алгоритма, с коротким контекстом на\n"
            "каждой строке.\n\n"
            "ВАЖНО. Блок после «ИТОГ:» читает аналитик, который НЕ видит\n"
            "твоих рассуждений по шагам. ЗАПРЕЩЕНЫ любые отсылки типа «как\n"
            "приведено выше», «описание в шагах выше», «см. шаг N»,\n"
            "«см. выше». Каждое описание алгоритма должно быть полностью\n"
            "воспроизведено внутри блока ИТОГ — со всеми шагами, формулами,\n"
            "порогами, метриками и именами показателей. Если в рассуждениях\n"
            "у тебя уже есть нужный кусок — перенеси его ЦЕЛИКОМ в финальный\n"
            "ответ, а не ссылайся на него."
        )

        steps = [
            "Прочитай каждый алгоритм из «# Отобранные алгоритмы» и отметь для себя шаги, метрики, источники.",
            "Прочитай «# Документы-первоисточники». Выпиши ВСЕ имена метрик/показателей/KPI/нормативных параметров, которые в них встречаются — даже если в базе знаний они не привязаны к алгоритму. Для каждого имени сразу зафиксируй: где встречается, какое значение/формула/порог указаны рядом.",
            "Прочитай «# Формат итогового ответа агента-аналитика» и прикинь, какие дополнительные сущности из документов надо вытащить, чтобы ответ в этом формате был полным (цифры, таблицы, списки этапов и т.п.).",
            "Сопоставь выписанные метрики и сущности с шагами алгоритмов (по имени, понятиям, ролям).",
            "Для КАЖДОГО алгоритма отдельно собери полное описание: введение (что делает и зачем), последовательность шагов, формулы/условия/пороги, связанные метрики (с деталями и целевыми значениями), ссылки на роли и системы.",
            "Убедись, что не опустил ни один шаг/формулу/число/исключение ни из базы, ни из документа, и что все имена метрик/показателей из документов присутствуют в итоговом тексте.",
            "Разверни аббревиатуры через список из «# Контекст».",
            "После последнего шага выведи «ИТОГ:» на новой строке и сразу за ним — детальные описания всех алгоритмов подряд, с подзаголовком «Метрики и показатели, упомянутые в документах» в конце каждого.",
        ]

        prompt = self._build_prompt(
            task=task,
            title="Детальное описание отобранных алгоритмов для аналитика",
            text=f"Инструкция (для справки, не для ответа на неё):\n{instruction}",
            steps=steps,
            context=context,
        )
        return self._call_free_text(prompt)

    # ==================================================================
    # Candidate fetch helpers
    # ==================================================================

    def _fetch_claim_candidates(
        self,
        cur,
        direction_id: int,
        embedding: list[float],
        threshold: float,
    ) -> list[dict]:
        cur.execute(
            "SELECT id, scope, short_description, detailed_description, "
            "       role_names, "
            "       1 - (short_description_embedding <=> %s::vector) AS similarity "
            "FROM rag_v2.claims "
            "WHERE direction_id = %s AND scope IN ('direction', 'role') "
            "  AND short_description_embedding IS NOT NULL "
            "ORDER BY short_description_embedding <=> %s::vector "
            "LIMIT %s",
            (embedding, direction_id, embedding, self.TOP_K),
        )
        candidates: list[dict] = []
        for r in cur.fetchall():
            row = dict(r)
            if row.pop("similarity") < threshold:
                continue
            candidates.append(row)
        return candidates

    def _fetch_term_candidates(
        self,
        cur,
        direction_id: int,
        embedding: list[float],
        threshold: float,
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
            (embedding, direction_id, embedding, self.TOP_K),
        )
        candidates: list[dict] = []
        for r in cur.fetchall():
            row = dict(r)
            if row.pop("similarity") < threshold:
                continue
            candidates.append(row)
        return candidates

    def _semantic_fetch(
        self,
        cur,
        *,
        table: str,
        direction_id: int,
        embedding: list[float],
        role_ids: list[int],
        threshold: float,
        extra_columns: str = "",
    ) -> list[dict]:
        columns = "id, name, short_description, detailed_description"
        if extra_columns:
            columns += ", " + extra_columns
        base_sql = (
            f"SELECT {columns}, "
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

    @staticmethod
    def _pair_ids_validator(
        *,
        claim_ids_key: str,
        term_ids_key: str,
        claim_index: dict,
        term_index: dict,
    ) -> Callable[[Any], None]:
        valid_claims = set(claim_index.keys())
        valid_terms = set(term_index.keys())

        def validate(data: Any) -> None:
            if not isinstance(data, dict):
                raise ValueError("expected JSON object")
            for key, valid in (
                (claim_ids_key, valid_claims),
                (term_ids_key, valid_terms),
            ):
                ids = data.get(key)
                if not isinstance(ids, list) or not all(
                    isinstance(i, int) for i in ids
                ):
                    raise ValueError(f"{key} must be a list of integers")
                for i in ids:
                    if i not in valid:
                        raise ValueError(
                            f"id {i} in {key} is not among the candidates"
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


def _extract_after_marker(text: str, marker: str) -> str | None:
    idx = text.rfind(marker)
    if idx < 0:
        return None
    tail = text[idx + len(marker):].strip()
    return tail or None


if __name__ == "__main__":
    import os
    import sys

    def _dummy_llm(prompt: str) -> str:  # noqa: ARG001
        raise NotImplementedError("replace with real LLM client")

    def _dummy_embed(text: str) -> list[float]:  # noqa: ARG001
        raise NotImplementedError("replace with real embedding client")

    dsn = os.environ.get("DATABASE_URL", "postgresql://localhost/knowledge")
    if len(sys.argv) < 6:
        print(
            "usage: python knowledge_extractor.py <direction_id> "
            "<role_query> <scenario> <instruction> <response_format>",
            file=sys.stderr,
        )
        sys.exit(2)

    extractor = KnowledgeExtractor(
        llm_call=_dummy_llm,
        embed=_dummy_embed,
        db_connection_string=dsn,
    )
    extractor.build_context(
        direction_id=int(sys.argv[1]),
        role_query=sys.argv[2],
        scenario=sys.argv[3],
        instruction=sys.argv[4],
        response_format=sys.argv[5],
    )
    print("=== organization_info ===")
    print(extractor.organization_info())
    print("=== direction_info ===")
    print(extractor.direction_info())
    print("=== situational_context ===")
    print(extractor.situational_context())
    print("=== related_algorithms ===")
    print(extractor.related_algorithms())
    print("=== related_metrics ===")
    print(extractor.related_metrics())
