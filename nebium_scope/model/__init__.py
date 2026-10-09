"""
Model adapter abstractions, activation hooks, and dummy fixtures.
"""

from nebium_scope.model.hooks import ActivationRecorder
from nebium_scope.model.dummy import DummyNebiumModel, DummyBlock
from nebium_scope.model.adapter import NebiumAdapter

__all__ = [
    "ActivationRecorder",
    "DummyNebiumModel",
    "DummyBlock",
    "NebiumAdapter",
]
