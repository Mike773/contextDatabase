from .base import Analyzer


class AlgorithmsAnalyzer(Analyzer):
    name = "algorithms"

    def run(self, document_id: int) -> None:
        raise NotImplementedError
