"""Backward-compatible re-exports; prefer sws.micro_draft.MicroDraftModel."""

from sws.micro_draft import MicroDraftModel, TinyFootprintPredictor

__all__ = ["MicroDraftModel", "TinyFootprintPredictor"]