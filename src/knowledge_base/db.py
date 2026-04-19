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

    def fetch_roles_by_direction(self, direction_id: int) -> list[dict]:
        with self.cursor() as cur:
            cur.execute(
                "SELECT id, name, short_description "
                "FROM rag_v2.roles WHERE direction_id = %s ORDER BY id",
                (direction_id,),
            )
            return [dict(r) for r in cur.fetchall()]

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
                "       1 - (short_description_embedding <=> %s) AS similarity "
                "FROM rag_v2.claims "
                "WHERE direction_id = %s AND scope = %s "
                "  AND short_description_embedding IS NOT NULL "
                "ORDER BY short_description_embedding <=> %s "
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
    ) -> int:
        with self.cursor() as cur:
            cur.execute(
                "INSERT INTO rag_v2.claims "
                "(direction_id, scope, short_description, detailed_description, "
                " document_ids, short_description_embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                (
                    direction_id,
                    scope,
                    short_description,
                    detailed_description,
                    [document_id],
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
