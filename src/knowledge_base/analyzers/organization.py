from .base import Analyzer


class OrganizationAnalyzer(Analyzer):
    name = "organization"

    def run(self, document_id: int) -> None:
        raise NotImplementedError
