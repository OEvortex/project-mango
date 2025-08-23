"""
Modular Layer (MoL) System for LLMs

A runtime system for dynamically combining transformer blocks from arbitrary
Large Language Models using adapters and routing mechanisms.

Now includes MergeKit-style model merging capabilities.
"""

# Core MoL Runtime components
from .core.mol_runtime import MoLRuntime
from .core.adapters import LinearAdapter, BottleneckAdapter
from .core.routers import SimpleRouter, TokenLevelRouter
from .core.block_extractor import BlockExtractor

# Merge methods (MergeKit-style)
from .merge_methods import SlerpMerge, TiesMerge, TaskArithmeticMerge, LinearMerge
from .merge_methods.base_merge import MergeConfig, BaseMergeMethod

# Configuration system
from .config import ConfigParser, ConfigValidator, MergeConfiguration

# CLI (when imported programmatically)
try:
    from .cli import merge_cli
except ImportError:
    merge_cli = None

__version__ = "0.2.0"
__all__ = [
    # Core MoL Runtime
    "MoLRuntime",
    "LinearAdapter", 
    "BottleneckAdapter",
    "SimpleRouter",
    "TokenLevelRouter",
    "BlockExtractor",
    
    # Merge Methods
    "SlerpMerge",
    "TiesMerge",
    "TaskArithmeticMerge",
    "LinearMerge",
    "BaseMergeMethod",
    "MergeConfig",
    
    # Configuration
    "ConfigParser",
    "ConfigValidator",
    "MergeConfiguration",
]