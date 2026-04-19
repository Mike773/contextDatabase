import re
from typing import Any, Callable

from .analyzers import ANALYZERS
from .clients import EmbeddingClient, LLMClient
from .structured import call_structured as _call_structured


def _expand_abbreviations(text: str, abbreviations: dict) -> str:
    if not abbreviations:
        return text
    for key in sorted(abbreviations.keys(), key=len, reverse=True):
        value = abbreviations[key]
        text = re.sub(rf"\b{re.escape(key)}\b", value, text)
    return text


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
        if len(argv) == 2 and argv[0] == "promote":
            stats = self.promote(int(argv[1]))
            import json
            print(json.dumps(stats, ensure_ascii=False, indent=2))
            return
        if len(argv) == 2:
            self.run(argv[0], int(argv[1]))
            return
        raise SystemExit(
            "usage:\n"
            f"  <analyzer> <document_id>   — analyzers: {sorted(ANALYZERS)}\n"
            "  promote <direction_id>     — promote staging to main tables"
        )

    def promote_roles(self, direction_id: int) -> dict:
        from .promotion import promote_roles
        return promote_roles(self, direction_id)

    def promote_metrics(self, direction_id: int) -> dict:
        from .promotion import promote_metrics
        return promote_metrics(self, direction_id)

    def promote_algorithms(self, direction_id: int) -> dict:
        from .promotion import promote_algorithms
        return promote_algorithms(self, direction_id)

    def promote(self, direction_id: int) -> dict:
        from .promotion import promote_all
        return promote_all(self, direction_id)

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

    def embed(self, text: str, abbreviations: dict | None = None) -> list[float]:
        if abbreviations:
            text = _expand_abbreviations(text, abbreviations)
        return self.embedding.embed(text)
