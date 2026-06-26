"""Speculative Weight Streaming — dynamic model recomposition from raw clay pieces."""

from sws.assembler import DynamicAssembler
from sws.cache import PredictiveCache
from sws.micro_draft import MicroDraftModel, TinyFootprintPredictor
from sws.ornith_config import ORNITH_MODEL_ID, OrnithMoEConfig
from sws.ornith_draft import OrnithMicroDraftModel, OrnithDraftSelector, propose_architecture_summary
from sws.store import NVMeWeightStore
from sws.streamer import SpeculativeWeightStreamer
from sws.types import AssembledSubgraph, ReassemblyBlueprint
from sws.verifier import Verifier

__all__ = [
    "NVMeWeightStore",
    "MicroDraftModel",
    "TinyFootprintPredictor",
    "OrnithMicroDraftModel",
    "OrnithDraftSelector",
    "OrnithMoEConfig",
    "ORNITH_MODEL_ID",
    "propose_architecture_summary",
    "DynamicAssembler",
    "PredictiveCache",
    "ReassemblyBlueprint",
    "AssembledSubgraph",
    "Verifier",
    "SpeculativeWeightStreamer",
]

__version__ = "0.3.0"