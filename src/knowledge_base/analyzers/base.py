from abc import ABC, abstractmethod
from typing import ClassVar, TYPE_CHECKING

if TYPE_CHECKING:
    from ..kb import KnowledgeBase


class Analyzer(ABC):
    name: ClassVar[str]

    def __init__(self, kb: "KnowledgeBase"):
        self.kb = kb

    @abstractmethod
    def run(self, document_id: int) -> None: ...
