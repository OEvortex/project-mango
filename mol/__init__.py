"""
Modular Layer (MoL) System for LLMs

A runtime system for dynamically combining transformer blocks from arbitrary
Large Language Models using adapters and routing mechanisms.
"""

from .core.mol_runtime import MoLRuntime
from .core.adapters import LinearAdapter, BottleneckAdapter
from .core.routers import SimpleRouter, TokenLevelRouter

__version__ = "0.1.0"
__all__ = [
    "MoLRuntime",
    "LinearAdapter", 
    "BottleneckAdapter",
    "SimpleRouter",
    "TokenLevelRouter",
]