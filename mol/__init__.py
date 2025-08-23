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

# Utilities
try:
    from .utils.hf_utils import HuggingFacePublisher, push_mol_to_hf
    HF_AVAILABLE = True
except ImportError:
    HuggingFacePublisher = None
    push_mol_to_hf = None
    HF_AVAILABLE = False

# SafeTensors support
try:
    from .utils.safetensors_utils import (
        SafeTensorsManager, save_model_safe, load_model_safe, 
        is_safetensors_available, safetensors_manager
    )
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SafeTensorsManager = None
    save_model_safe = None
    load_model_safe = None
    is_safetensors_available = lambda: False
    safetensors_manager = None
    SAFETENSORS_AVAILABLE = False

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
    
    # Hugging Face Integration (optional)
    "HuggingFacePublisher",
    "push_mol_to_hf",
    "HF_AVAILABLE",
    
    # SafeTensors Integration (optional)
    "SafeTensorsManager",
    "save_model_safe",
    "load_model_safe",
    "is_safetensors_available",
    "safetensors_manager",
    "SAFETENSORS_AVAILABLE",
]