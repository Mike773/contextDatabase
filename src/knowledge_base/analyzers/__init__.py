from .algorithms import AlgorithmsAnalyzer
from .analysis_plan import AnalysisPlanAnalyzer
from .base import Analyzer
from .direction import DirectionAnalyzer
from .metrics import MetricsAnalyzer
from .organization import OrganizationAnalyzer
from .roles import RolesAnalyzer
from .summary import SummaryAnalyzer

ANALYZERS: dict[str, type[Analyzer]] = {
    SummaryAnalyzer.name: SummaryAnalyzer,
    AnalysisPlanAnalyzer.name: AnalysisPlanAnalyzer,
    OrganizationAnalyzer.name: OrganizationAnalyzer,
    DirectionAnalyzer.name: DirectionAnalyzer,
    RolesAnalyzer.name: RolesAnalyzer,
    MetricsAnalyzer.name: MetricsAnalyzer,
    AlgorithmsAnalyzer.name: AlgorithmsAnalyzer,
}

__all__ = [
    "ANALYZERS",
    "Analyzer",
    "AlgorithmsAnalyzer",
    "AnalysisPlanAnalyzer",
    "DirectionAnalyzer",
    "MetricsAnalyzer",
    "OrganizationAnalyzer",
    "RolesAnalyzer",
    "SummaryAnalyzer",
]
