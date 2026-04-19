from .base import Analyzer


class AnalysisPlanAnalyzer(Analyzer):
    name = "analysis_plan"

    def run(self, document_id: int) -> None:
        raise NotImplementedError
