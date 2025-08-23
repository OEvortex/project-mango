"""
Adapter layers for dimension matching between different models.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class BaseAdapter(nn.Module, ABC):
    """Base class for all adapters."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the adapter."""
        pass


class LinearAdapter(BaseAdapter):
    """
    Linear adapter for dimension matching.
    
    Simple linear projection with optional LayerNorm and residual connection.
    Supports identity initialization for warm start.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int,
        use_layer_norm: bool = True,
        use_residual: bool = True,
        init_identity: bool = True
    ):
        super().__init__(input_dim, output_dim)
        
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual and (input_dim == output_dim)
        
        # Linear projection
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        
        # Initialize for identity mapping when possible
        if init_identity:
            self._init_identity()
    
    def _init_identity(self):
        """Initialize adapter to approximate identity mapping."""
        try:
            if self.input_dim == self.output_dim:
                if self.use_residual:
                    # If using residual connection, initialize projection to zero
                    # so that output = 0 + x = x (identity)
                    nn.init.zeros_(self.projection.weight)
                else:
                    # Perfect identity initialization without residual
                    nn.init.eye_(self.projection.weight)
            else:
                # For non-matching dimensions, use small random weights
                std = 0.01 / max(self.input_dim, self.output_dim) ** 0.5
                nn.init.normal_(self.projection.weight, mean=0.0, std=std)
        except Exception as e:
            logger.warning(f"Identity initialization failed: {e}. Using default initialization.")
            # Fallback to default initialization
            nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through linear adapter."""
        original_x = x
        
        # Apply linear projection
        x = self.projection(x)
        
        # Apply layer normalization
        if self.use_layer_norm:
            x = self.layer_norm(x)
        
        # Add residual connection if dimensions match
        if self.use_residual:
            x = x + original_x
        
        return x


class BottleneckAdapter(BaseAdapter):
    """
    Bottleneck adapter for efficient dimension matching.
    
    Uses down-projection -> activation -> up-projection architecture
    to reduce parameter count while maintaining expressiveness.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bottleneck_dim: Optional[int] = None,
        activation: str = "gelu",
        init_identity: bool = True,
        dropout_rate: float = 0.1
    ):
        super().__init__(input_dim, output_dim)
        
        # Default bottleneck dimension is 1/8 of max(input, output) dim, min 64
        if bottleneck_dim is None:
            bottleneck_dim = max(64, max(input_dim, output_dim) // 8)
        
        self.bottleneck_dim = bottleneck_dim
        
        # Down projection
        self.down_proj = nn.Linear(input_dim, bottleneck_dim, bias=True)
        
        # Activation function
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Up projection
        self.up_proj = nn.Linear(bottleneck_dim, output_dim, bias=False)
        
        # Layer norm for output
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # Residual connection when dimensions match
        self.use_residual = (input_dim == output_dim)
        
        # Initialize for identity-like behavior
        if init_identity:
            self._init_identity()
    
    def _init_identity(self):
        """Initialize adapter to approximate identity mapping."""
        # Initialize down projection with small weights
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.down_proj.bias)
        
        # Initialize up projection with very small weights to start near identity
        nn.init.zeros_(self.up_proj.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through bottleneck adapter."""
        original_x = x
        
        # Check for NaN or inf in input
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("NaN or inf detected in adapter input")
            if self.use_residual:
                return original_x
            else:
                return torch.zeros_like(x)
        
        # Down projection
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Up projection
        x = self.up_proj(x)
        
        # Apply layer norm before residual
        x = self.layer_norm(x)
        
        # Add residual connection if dimensions match
        if self.use_residual:
            # Scale the adapter output for better training stability
            adapter_scale = 0.1  # Small scaling factor
            x = original_x + adapter_scale * x
        
        # Final check for valid output
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("NaN or inf detected in adapter output, returning input")
            return original_x if self.use_residual else torch.zeros_like(original_x)
        
        return x


def create_adapter(
    adapter_type: str,
    input_dim: int,
    output_dim: int,
    **kwargs
) -> BaseAdapter:
    """Factory function to create adapters."""
    adapter_type = adapter_type.lower()
    
    if adapter_type == "linear":
        return LinearAdapter(input_dim, output_dim, **kwargs)
    elif adapter_type == "bottleneck":
        return BottleneckAdapter(input_dim, output_dim, **kwargs)
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")