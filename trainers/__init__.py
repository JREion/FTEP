"""Trainer registry for the four supported FTEP experiments."""

# Dataset registration is kept in train.py so importing this package remains cheap.
from .coop import CoOp
from .dept_coop import DePT
from .mmrl import MMRL
from .ftep_mmrl import FTEP_MMRL

__all__ = ["CoOp", "DePT", "MMRL", "FTEP_MMRL"]
