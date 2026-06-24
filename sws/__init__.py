"""Speculative Weight Streaming — prediction-driven MoE weight cache."""

from sws.cache import PredictiveCache
from sws.predictor import TinyFootprintPredictor
from sws.store import NVMeWeightStore
from sws.streamer import SpeculativeWeightStreamer

__all__ = [
    "NVMeWeightStore",
    "PredictiveCache",
    "TinyFootprintPredictor",
    "SpeculativeWeightStreamer",
]

__version__ = "0.1.0"