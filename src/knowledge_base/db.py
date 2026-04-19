import psycopg2
import psycopg2.extras


class DB:
    def __init__(self, dsn: str):
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = False

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
