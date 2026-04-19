import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector


class DB:
    def __init__(self, dsn: str):
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = False
        register_vector(self._conn)

    def cursor(self):
        return self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        self._conn.close()

    def fetch_document(self, document_id: int) -> dict | None:
        with self.cursor() as cur:
            cur.execute(
                "SELECT id, direction_id, title, text, should_chunk, chunk_separator, "
                "summary, unclear_items, analysis_plan "
                "FROM rag_v2.documents WHERE id = %s",
                (document_id,),
            )
            row = cur.fetchone()
        return dict(row) if row else None

    def fetch_direction(self, direction_id: int) -> dict | None:
        with self.cursor() as cur:
            cur.execute(
                "SELECT id, short_name, full_name, general_info, abbreviations "
                "FROM rag_v2.directions WHERE id = %s",
                (direction_id,),
            )
            row = cur.fetchone()
        return dict(row) if row else None

    def update_document_summary(
        self, document_id: int, summary: str, summary_embedding: list[float]
    ) -> None:
        with self.cursor() as cur:
            cur.execute(
                "UPDATE rag_v2.documents "
                "SET summary = %s, summary_embedding = %s WHERE id = %s",
                (summary, summary_embedding, document_id),
            )
        self.commit()

    def fetch_metrics_by_direction(self, direction_id: int) -> list[dict]:
        with self.cursor() as cur:
            cur.execute(
                "SELECT id, name, short_description "
                "FROM rag_v2.metrics WHERE direction_id = %s ORDER BY id",
                (direction_id,),
            )
            return [dict(r) for r in cur.fetchall()]

    def fetch_algorithms_by_direction(self, direction_id: int) -> list[dict]:
        with self.cursor() as cur:
            cur.execute(
                "SELECT id, name, short_description "
                "FROM rag_v2.algorithms WHERE direction_id = %s ORDER BY id",
                (direction_id,),
            )
            return [dict(r) for r in cur.fetchall()]

    def fetch_roles_by_direction(self, direction_id: int) -> list[dict]:
        with self.cursor() as cur:
            cur.execute(
                "SELECT id, name, short_description, detailed_description "
                "FROM rag_v2.roles WHERE direction_id = %s ORDER BY id",
                (direction_id,),
            )
            return [dict(r) for r in cur.fetchall()]

    def insert_extraction(
        self,
        *,
        direction_id: int,
        document_id: int,
        entity_type: str,
        name: str | None = None,
        description: str | None = None,
        quote: str | None = None,
        alternative_names: list[str] | None = None,
        related_role_names: list[str] | None = None,
        related_metric_names: list[str] | None = None,
    ) -> int:
        with self.cursor() as cur:
            cur.execute(
                "INSERT INTO rag_v2.extractions "
                "(direction_id, document_id, entity_type, name, description, "
                " quote, alternative_names, related_role_names, "
                " related_metric_names) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
                (
                    direction_id,
                    document_id,
                    entity_type,
                    name,
                    description,
                    quote,
                    alternative_names or [],
                    related_role_names or [],
                    related_metric_names or [],
                ),
            )
            new_id = cur.fetchone()["id"]
        self.commit()
        return new_id

    def update_document_analysis_plan(
        self, document_id: int, plan: dict
    ) -> None:
        with self.cursor() as cur:
            cur.execute(
                "UPDATE rag_v2.documents SET analysis_plan = %s WHERE id = %s",
                (psycopg2.extras.Json(plan), document_id),
            )
        self.commit()

    def fetch_terms_by_scope(
        self, direction_id: int, scope: str
    ) -> list[dict]:
        with self.cursor() as cur:
            cur.execute(
                "SELECT id, name FROM rag_v2.terms "
                "WHERE direction_id = %s AND scope = %s ORDER BY id",
                (direction_id, scope),
            )
            return [dict(r) for r in cur.fetchall()]

    def find_term_by_name(
        self, direction_id: int, scope: str, name: str
    ) -> int | None:
        with self.cursor() as cur:
            cur.execute(
                "SELECT id FROM rag_v2.terms "
                "WHERE direction_id = %s AND scope = %s "
                "AND LOWER(name) = LOWER(%s) LIMIT 1",
                (direction_id, scope, name),
            )
            row = cur.fetchone()
            return row["id"] if row else None

    def append_term_quote(
        self, term_id: int, document_id: int, quote: str
    ) -> None:
        entry = [{"document_id": document_id, "quote": quote}]
        with self.cursor() as cur:
            cur.execute(
                "UPDATE rag_v2.terms SET quotes = quotes || %s WHERE id = %s",
                (psycopg2.extras.Json(entry), term_id),
            )
        self.commit()

    def insert_term(
        self,
        *,
        direction_id: int,
        scope: str,
        name: str,
        short_description: str | None,
        detailed_description: str | None,
        document_id: int,
        quote: str,
        name_embedding: list[float],
        short_description_embedding: list[float] | None,
    ) -> int:
        quotes = [{"document_id": document_id, "quote": quote}]
        with self.cursor() as cur:
            cur.execute(
                "INSERT INTO rag_v2.terms "
                "(direction_id, scope, name, short_description, "
                " detailed_description, quotes, name_embedding, "
                " short_description_embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
                (
                    direction_id,
                    scope,
                    name,
                    short_description,
                    detailed_description,
                    psycopg2.extras.Json(quotes),
                    name_embedding,
                    short_description_embedding,
                ),
            )
            new_id = cur.fetchone()["id"]
        self.commit()
        return new_id

    def find_similar_claims(
        self,
        direction_id: int,
        scope: str,
        embedding: list[float],
        top_k: int = 3,
        threshold: float = 0.80,
    ) -> list[dict]:
        with self.cursor() as cur:
            cur.execute(
                "SELECT id, short_description, detailed_description, "
                "       1 - (short_description_embedding <=> %s::vector) AS similarity "
                "FROM rag_v2.claims "
                "WHERE direction_id = %s AND scope = %s "
                "  AND short_description_embedding IS NOT NULL "
                "ORDER BY short_description_embedding <=> %s::vector "
                "LIMIT %s",
                (embedding, direction_id, scope, embedding, top_k),
            )
            rows = [dict(r) for r in cur.fetchall()]
        return [r for r in rows if r["similarity"] >= threshold]

    def insert_claim(
        self,
        *,
        direction_id: int,
        scope: str,
        short_description: str,
        detailed_description: str,
        document_id: int,
        short_description_embedding: list[float],
        role_names: list[str] | None = None,
    ) -> int:
        with self.cursor() as cur:
            cur.execute(
                "INSERT INTO rag_v2.claims "
                "(direction_id, scope, short_description, detailed_description, "
                " document_ids, role_names, short_description_embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
                (
                    direction_id,
                    scope,
                    short_description,
                    detailed_description,
                    [document_id],
                    role_names or [],
                    short_description_embedding,
                ),
            )
            new_id = cur.fetchone()["id"]
        self.commit()
        return new_id

    def append_claim_document(
        self, claim_id: int, document_id: int
    ) -> None:
        with self.cursor() as cur:
            cur.execute(
                "UPDATE rag_v2.claims "
                "SET document_ids = array_append(document_ids, %s) "
                "WHERE id = %s AND NOT (%s = ANY(document_ids))",
                (document_id, claim_id, document_id),
            )
        self.commit()

    # ------------------------------------------------------------------
    # Promotion: extractions → rag_v2.{roles,metrics,algorithms}
    # ------------------------------------------------------------------

    def fetch_pending_extractions(
        self, direction_id: int, entity_type: str
    ) -> list[dict]:
        with self.cursor() as cur:
            cur.execute(
                "SELECT id, direction_id, document_id, entity_type, name, "
                "       description, quote, alternative_names, "
                "       related_role_names, related_metric_names "
                "FROM rag_v2.extractions "
                "WHERE direction_id = %s AND entity_type = %s "
                "  AND status = 'pending' ORDER BY id",
                (direction_id, entity_type),
            )
            return [dict(r) for r in cur.fetchall()]

    def mark_extraction_loaded(self, extraction_id: int) -> None:
        with self.cursor() as cur:
            cur.execute(
                "UPDATE rag_v2.extractions SET status = 'loaded' WHERE id = %s",
                (extraction_id,),
            )
        self.commit()

    # --- roles -----------------------------------------------------------

    def find_role_by_name(
        self, direction_id: int, name: str
    ) -> int | None:
        with self.cursor() as cur:
            cur.execute(
                "SELECT id FROM rag_v2.roles "
                "WHERE direction_id = %s AND LOWER(name) = LOWER(%s) LIMIT 1",
                (direction_id, name),
            )
            row = cur.fetchone()
            return row["id"] if row else None

    def find_role_by_any_name(
        self, direction_id: int, name: str
    ) -> int | None:
        with self.cursor() as cur:
            cur.execute(
                "SELECT id FROM rag_v2.roles "
                "WHERE direction_id = %s "
                "  AND (LOWER(name) = LOWER(%s) "
                "       OR EXISTS (SELECT 1 FROM unnest(alternative_names) alt "
                "                  WHERE LOWER(alt) = LOWER(%s))) "
                "LIMIT 1",
                (direction_id, name, name),
            )
            row = cur.fetchone()
            return row["id"] if row else None

    def insert_role(
        self,
        *,
        direction_id: int,
        name: str,
        detailed_description: str,
        alternative_names: list[str],
        document_id: int,
        quote: str,
        name_embedding: list[float],
    ) -> int:
        quotes = [{"document_id": document_id, "quote": quote}]
        with self.cursor() as cur:
            cur.execute(
                "INSERT INTO rag_v2.roles "
                "(direction_id, name, detailed_description, quotes, "
                " alternative_names, name_embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                (
                    direction_id,
                    name,
                    detailed_description,
                    psycopg2.extras.Json(quotes),
                    alternative_names,
                    name_embedding,
                ),
            )
            new_id = cur.fetchone()["id"]
        self.commit()
        return new_id

    def merge_role(
        self,
        role_id: int,
        *,
        alternative_names: list[str],
        document_id: int,
        quote: str,
    ) -> None:
        new_quote = [{"document_id": document_id, "quote": quote}]
        with self.cursor() as cur:
            cur.execute(
                "UPDATE rag_v2.roles "
                "SET quotes = quotes || %s, "
                "    alternative_names = ARRAY(SELECT DISTINCT unnest("
                "        alternative_names || %s::text[])) "
                "WHERE id = %s",
                (
                    psycopg2.extras.Json(new_quote),
                    alternative_names,
                    role_id,
                ),
            )
        self.commit()

    # --- metrics ---------------------------------------------------------

    def find_similar_metrics(
        self,
        direction_id: int,
        embedding: list[float],
        top_k: int = 3,
        threshold: float = 0.80,
    ) -> list[dict]:
        with self.cursor() as cur:
            cur.execute(
                "SELECT id, name, detailed_description, "
                "       1 - (short_description_embedding <=> %s::vector) AS similarity "
                "FROM rag_v2.metrics "
                "WHERE direction_id = %s "
                "  AND short_description_embedding IS NOT NULL "
                "ORDER BY short_description_embedding <=> %s::vector "
                "LIMIT %s",
                (embedding, direction_id, embedding, top_k),
            )
            rows = [dict(r) for r in cur.fetchall()]
        return [r for r in rows if r["similarity"] >= threshold]

    def insert_metric(
        self,
        *,
        direction_id: int,
        name: str,
        detailed_description: str,
        document_id: int,
        role_ids: list[int],
        name_embedding: list[float],
        description_embedding: list[float],
    ) -> int:
        with self.cursor() as cur:
            cur.execute(
                "INSERT INTO rag_v2.metrics "
                "(direction_id, name, detailed_description, document_ids, "
                " role_ids, name_embedding, short_description_embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
                (
                    direction_id,
                    name,
                    detailed_description,
                    [document_id],
                    role_ids,
                    name_embedding,
                    description_embedding,
                ),
            )
            new_id = cur.fetchone()["id"]
        self.commit()
        return new_id

    def merge_metric(
        self, metric_id: int, *, document_id: int, role_ids: list[int]
    ) -> None:
        with self.cursor() as cur:
            cur.execute(
                "UPDATE rag_v2.metrics "
                "SET document_ids = ARRAY(SELECT DISTINCT unnest("
                "        document_ids || ARRAY[%s]::bigint[])), "
                "    role_ids = ARRAY(SELECT DISTINCT unnest("
                "        role_ids || %s::bigint[])) "
                "WHERE id = %s",
                (document_id, role_ids, metric_id),
            )
        self.commit()

    # --- algorithms ------------------------------------------------------

    def find_similar_algorithms(
        self,
        direction_id: int,
        embedding: list[float],
        top_k: int = 3,
        threshold: float = 0.80,
    ) -> list[dict]:
        with self.cursor() as cur:
            cur.execute(
                "SELECT id, name, detailed_description, "
                "       1 - (short_description_embedding <=> %s::vector) AS similarity "
                "FROM rag_v2.algorithms "
                "WHERE direction_id = %s "
                "  AND short_description_embedding IS NOT NULL "
                "ORDER BY short_description_embedding <=> %s::vector "
                "LIMIT %s",
                (embedding, direction_id, embedding, top_k),
            )
            rows = [dict(r) for r in cur.fetchall()]
        return [r for r in rows if r["similarity"] >= threshold]

    def insert_algorithm(
        self,
        *,
        direction_id: int,
        name: str,
        detailed_description: str,
        document_id: int,
        quote: str,
        role_ids: list[int],
        metric_ids: list[int],
        name_embedding: list[float],
        description_embedding: list[float],
    ) -> int:
        quotes = [{"document_id": document_id, "quote": quote}]
        with self.cursor() as cur:
            cur.execute(
                "INSERT INTO rag_v2.algorithms "
                "(direction_id, name, detailed_description, quotes, "
                " document_ids, role_ids, metric_ids, name_embedding, "
                " short_description_embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
                (
                    direction_id,
                    name,
                    detailed_description,
                    psycopg2.extras.Json(quotes),
                    [document_id],
                    role_ids,
                    metric_ids,
                    name_embedding,
                    description_embedding,
                ),
            )
            new_id = cur.fetchone()["id"]
        self.commit()
        return new_id

    def merge_algorithm(
        self,
        algorithm_id: int,
        *,
        document_id: int,
        quote: str,
        role_ids: list[int],
        metric_ids: list[int],
    ) -> None:
        new_quote = [{"document_id": document_id, "quote": quote}]
        with self.cursor() as cur:
            cur.execute(
                "UPDATE rag_v2.algorithms "
                "SET quotes = quotes || %s, "
                "    document_ids = ARRAY(SELECT DISTINCT unnest("
                "        document_ids || ARRAY[%s]::bigint[])), "
                "    role_ids = ARRAY(SELECT DISTINCT unnest("
                "        role_ids || %s::bigint[])), "
                "    metric_ids = ARRAY(SELECT DISTINCT unnest("
                "        metric_ids || %s::bigint[])) "
                "WHERE id = %s",
                (
                    psycopg2.extras.Json(new_quote),
                    document_id,
                    role_ids,
                    metric_ids,
                    algorithm_id,
                ),
            )
        self.commit()
