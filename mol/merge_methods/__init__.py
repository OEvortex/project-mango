"""
Merge methods for combining models similar to MergeKit.
"""

from .base_merge import BaseMergeMethod
from .slerp import SlerpMerge
from .ties import TiesMerge
from .task_arithmetic import TaskArithmeticMerge
from .linear import LinearMerge

__all__ = [
    "BaseMergeMethod",
    "SlerpMerge", 
    "TiesMerge",
    "TaskArithmeticMerge",
    "LinearMerge",
]