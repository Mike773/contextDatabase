import os
import sys

from knowledge_base import EmbeddingClient, KnowledgeBase, LLMClient


class _DummyLLM:
    def complete(self, prompt: str) -> str:
        raise NotImplementedError("replace with real LLM client")


class _DummyEmbedding:
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError("replace with real embedding client")


def make_kb() -> KnowledgeBase:
    dsn = os.environ.get("DATABASE_URL", "postgresql://localhost/knowledge")
    llm: LLMClient = _DummyLLM()
    embedding: EmbeddingClient = _DummyEmbedding()
    return KnowledgeBase(dsn=dsn, llm=llm, embedding=embedding)


if __name__ == "__main__":
    make_kb().cli(sys.argv[1:])
