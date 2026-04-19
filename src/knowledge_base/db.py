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
