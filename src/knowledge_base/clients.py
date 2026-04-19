from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...


@runtime_checkable
class EmbeddingClient(Protocol):
    def embed(self, text: str) -> list[float]: ...
