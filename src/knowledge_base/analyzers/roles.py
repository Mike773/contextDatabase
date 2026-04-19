from .base import Analyzer


class RolesAnalyzer(Analyzer):
    name = "roles"

    def run(self, document_id: int) -> None:
        raise NotImplementedError
