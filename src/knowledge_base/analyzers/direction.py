from .base import Analyzer


class DirectionAnalyzer(Analyzer):
    name = "direction"

    def run(self, document_id: int) -> None:
        raise NotImplementedError
