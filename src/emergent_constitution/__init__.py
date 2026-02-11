"""The Emergent Constitution — agent-based political economy simulation."""

from emergent_constitution.config import SimulationConfig
from emergent_constitution.lead import Lead
from emergent_constitution.models.history import SimulationOutput

__version__ = "0.1.0"

__all__ = ["Lead", "SimulationConfig", "SimulationOutput", "__version__"]
