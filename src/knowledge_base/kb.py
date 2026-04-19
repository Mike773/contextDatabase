from typing import Any, Callable

from .analyzers import ANALYZERS
from .clients import EmbeddingClient, LLMClient
from .structured import call_structured as _call_structured


class KnowledgeBase:
    def __init__(self, dsn: str, llm: LLMClient, embedding: EmbeddingClient):
        from .db import DB

        self.db = DB(dsn)
        self.llm = llm
        self.embedding = embedding

    def run(self, analyzer_name: str, document_id: int) -> None:
        if analyzer_name not in ANALYZERS:
            raise KeyError(
                f"unknown analyzer {analyzer_name!r}; known: {sorted(ANALYZERS)}"
            )
        ANALYZERS[analyzer_name](self).run(document_id)

    def cli(self, argv: list[str]) -> None:
        if len(argv) != 2:
            raise SystemExit(
                f"usage: <analyzer> <document_id>\nanalyzers: {sorted(ANALYZERS)}"
            )
        self.run(argv[0], int(argv[1]))

    def call_structured(
        self,
        prompt: str,
        *,
        validate: Callable[[Any], None] | None = None,
        max_retries: int = 3,
    ) -> Any:
        return _call_structured(
            self.llm, prompt, validate=validate, max_retries=max_retries
        )
