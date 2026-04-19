from .base import Analyzer


class MetricsAnalyzer(Analyzer):
    name = "metrics"

    def run(self, document_id: int) -> None:
        raise NotImplementedError
