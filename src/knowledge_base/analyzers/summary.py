from .base import Analyzer


class SummaryAnalyzer(Analyzer):
    name = "summary"

    def run(self, document_id: int) -> None:
        raise NotImplementedError
