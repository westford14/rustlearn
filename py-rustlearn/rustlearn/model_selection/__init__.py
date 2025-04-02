"""Expose the model selection API."""

from rustlearn.model_selection.simple import SimpleTrainTestSplit
from rustlearn.model_selection.types import TrainTestSplitReturn


__all__ = [
    # Train test splitters
    "SimpleTrainTestSplit",
    # Return objects
    TrainTestSplitReturn,
]
