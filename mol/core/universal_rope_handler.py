"""
Universal RoPE Handler for Transformer Models

This module provides dynamic RoPE (Rotary Position Embedding) detection and generation
for any transformer model, eliminating hardcoded RoPE parameters like theta values
and head dimensions. Uses pure introspection of model configurations and implementations.
"""

import torch
import torch.nn as nn
import math
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RoPEConfig:
    """Configuration for RoPE embeddings detected from model."""
    theta: float
    head_dim: int
    max_position_embeddings: int
    scaling_factor: float = 1.0
    rope_type: str = "default"  # default, linear, dynamic, etc.
    partial_rotary_factor: float = 1.0
    
    def __post_init__(self):
        """Validate RoPE configuration."""
        if self.head_dim % 2 != 0:
            logger.warning(f"Head dimension {self.head_dim} is odd, adjusting for RoPE")
            self.head_dim = self.head_dim - 1
        
        if self.theta <= 0:
            logger.warning(f"Invalid theta {self.theta}, using default 10000.0")
            self.theta = 10000.0


@dataclass
class RoPEInfo:
    """Information about a model's RoPE implementation."""
    has_rope: bool
    config: Optional[RoPEConfig]
    implementation_path: Optional[str]  # Path to RoPE module in model
    uses_cos_sin_cache: bool
    supports_dynamic_scaling: bool
    rope_module: Optional[nn.Module] = None


class UniversalRoPEHandler:
    """
    Universal handler for RoPE (Rotary Position Embeddings) across all transformer models.
    
    Dynamically detects RoPE configuration and implementation without any hardcoded
    model-specific logic. Works by introspecting model configurations and RoPE modules.
    """
    
    # Common attribute names for RoPE-related parameters (discovery patterns)
    ROPE_CONFIG_ATTRIBUTES = [
        'rope_theta', 'theta', 'rotary_emb_base', 'rope_base',
        'head_dim', 'rotary_dim', 'rotary_ndims',
        'max_position_embeddings', 'max_seq_length', 'n_positions',
        'rope_scaling', 'scaling_factor', 'scaling_type',
        'partial_rotary_factor', 'rotary_percentage'
    ]
    
    # Common paths where RoPE modules might be found
    ROPE_MODULE_PATHS = [
        'rotary_emb',
        'rope',
        'self_attn.rotary_emb',
        'attention.rotary_emb',
        'rotary_pos_emb',
        'pos_emb',
        'rope_emb'
    ]
    
    def __init__(self):
        """Initialize universal RoPE handler."""
        self.rope_cache: Dict[str, RoPEInfo] = {}
    
    def detect_rope_info(self, model: nn.Module, config: Any, model_name: str = None) -> RoPEInfo:
        """
        Detect RoPE information from model and configuration.
        
        Args:
            model: The transformer model
            config: Model configuration object
            model_name: Optional model name for caching
            
        Returns:
            RoPEInfo object with detected RoPE information
        """
        if model_name and model_name in self.rope_cache:
            return self.rope_cache[model_name]
        
        if model_name is None:
            model_name = getattr(config, 'name_or_path', model.__class__.__name__)
        
        logger.info(f"Detecting RoPE info for {model_name}")
        
        # Step 1: Detect RoPE configuration from config
        rope_config = self._detect_rope_config(config)
        
        # Step 2: Find RoPE module in model
        rope_module, rope_path = self._find_rope_module(model)
        
        # Step 3: Validate and enhance config with module info
        if rope_module and rope_config:
            rope_config = self._enhance_config_from_module(rope_config, rope_module)
        elif rope_module and not rope_config:
            # Try to extract config from module
            rope_config = self._extract_config_from_module(rope_module, config)
        
        # Step 4: Check capabilities
        has_rope = rope_config is not None or rope_module is not None
        uses_cos_sin_cache = self._check_cos_sin_cache(rope_module) if rope_module else False
        supports_dynamic_scaling = self._check_dynamic_scaling(config, rope_module)
        
        rope_info = RoPEInfo(
            has_rope=has_rope,
            config=rope_config,
            implementation_path=rope_path,
            uses_cos_sin_cache=uses_cos_sin_cache,
            supports_dynamic_scaling=supports_dynamic_scaling,
            rope_module=rope_module
        )
        
        # Cache result
        if model_name:
            self.rope_cache[model_name] = rope_info
        
        logger.info(f"RoPE detection complete for {model_name}: has_rope={has_rope}")
        if rope_config:
            logger.debug(f"RoPE config: theta={rope_config.theta}, head_dim={rope_config.head_dim}")
        
        return rope_info
    
    def _detect_rope_config(self, config: Any) -> Optional[RoPEConfig]:
        """Detect RoPE configuration from model config."""
        if config is None:
            return None
        
        # First, check if this model type actually uses RoPE
        model_type = getattr(config, 'model_type', '').lower()
        
        # Models that definitely DON'T use RoPE
        non_rope_models = {
            'gpt2', 'bert', 'roberta', 'distilbert', 'albert', 'electra',
            'deberta', 't5', 'bart', 'pegasus', 'mbart', 'blenderbot',
            'dialogpt'  # DialoGPT uses absolute position embeddings
        }
        
        if model_type in non_rope_models:
            logger.debug(f"Model type {model_type} does not use RoPE")
            return None
        
        # Extract RoPE-related attributes dynamically
        rope_attrs = {}
        for attr_name in self.ROPE_CONFIG_ATTRIBUTES:
            if hasattr(config, attr_name):
                rope_attrs[attr_name] = getattr(config, attr_name)
        
        # Check for explicit RoPE indicators
        has_explicit_rope = any(attr in rope_attrs for attr in [
            'rope_theta', 'rotary_emb_base', 'rope_base', 'rope_scaling'
        ])
        
        if not has_explicit_rope and not rope_attrs:
            logger.debug("No RoPE attributes found in config")
            return None
        
        # Models that definitely DO use RoPE
        rope_models = {
            'llama', 'mistral', 'qwen', 'qwen2', 'phi', 'gemma', 'yi',
            'codellama', 'falcon', 'mpt', 'gpt_neox', 'pythia'
        }
        
        # Only proceed if we have explicit RoPE config OR it's a known RoPE model
        if not has_explicit_rope and model_type not in rope_models:
            logger.debug(f"Model type {model_type} is not a known RoPE model and has no explicit RoPE config")
            return None
        
        logger.debug(f"Found RoPE attributes: {rope_attrs}")
        
        # Determine theta (base frequency)
        theta = (rope_attrs.get('rope_theta') or 
                rope_attrs.get('theta') or 
                rope_attrs.get('rotary_emb_base') or 
                rope_attrs.get('rope_base') or 
                10000.0)  # Standard default
        
        # Determine head dimension
        head_dim = rope_attrs.get('head_dim')
        if head_dim is None:
            # Calculate from hidden_size and num_attention_heads
            hidden_size = getattr(config, 'hidden_size', getattr(config, 'd_model', None))
            num_heads = getattr(config, 'num_attention_heads', getattr(config, 'num_heads', None))
            if hidden_size and num_heads:
                head_dim = hidden_size // num_heads
                logger.debug(f"Calculated head_dim: {head_dim} = {hidden_size} / {num_heads}")
        
        if head_dim is None:
            logger.warning("Could not determine head dimension for RoPE")
            return None
        
        # Determine max position embeddings
        max_pos = (rope_attrs.get('max_position_embeddings') or 
                  rope_attrs.get('max_seq_length') or 
                  rope_attrs.get('n_positions') or 
                  2048)  # Common default
        
        # Handle scaling
        scaling_factor = 1.0
        rope_type = "default"
        if 'rope_scaling' in rope_attrs:
            scaling_info = rope_attrs['rope_scaling']
            if isinstance(scaling_info, dict):
                scaling_factor = scaling_info.get('factor', 1.0)
                rope_type = scaling_info.get('type', 'default')
        elif 'scaling_factor' in rope_attrs:
            scaling_factor = rope_attrs['scaling_factor']
        
        # Partial rotary factor
        partial_rotary_factor = (rope_attrs.get('partial_rotary_factor') or 
                               rope_attrs.get('rotary_percentage') or 
                               1.0)
        
        return RoPEConfig(
            theta=float(theta),
            head_dim=int(head_dim),
            max_position_embeddings=int(max_pos),
            scaling_factor=float(scaling_factor),
            rope_type=rope_type,
            partial_rotary_factor=float(partial_rotary_factor)
        )
    
    def _find_rope_module(self, model: nn.Module) -> Tuple[Optional[nn.Module], Optional[str]]:
        """Find RoPE module in model using introspection."""
        # First, try common paths
        for path in self.ROPE_MODULE_PATHS:
            try:
                module = self._navigate_to_module(model, path)
                if module is not None and self._is_rope_module(module):
                    logger.debug(f"Found RoPE module at path: {path}")
                    return module, path
            except (AttributeError, ValueError):
                continue
        
        # Second, search through all modules for RoPE-like components
        for name, module in model.named_modules():
            if self._is_rope_module(module):
                logger.debug(f"Found RoPE module via search: {name}")
                return module, name
        
        # Third, look for RoPE in attention layers
        attention_modules = self._find_attention_modules(model)
        for attn_name, attn_module in attention_modules:
            for name, module in attn_module.named_modules():
                if self._is_rope_module(module):
                    full_path = f"{attn_name}.{name}" if name else attn_name
                    logger.debug(f"Found RoPE module in attention: {full_path}")
                    return module, full_path
        
        logger.debug("No RoPE module found")
        return None, None
    
    def _navigate_to_module(self, model: nn.Module, path: str) -> Optional[nn.Module]:
        """Navigate to a module using dot notation."""
        attrs = path.split('.')
        current = model
        
        for attr in attrs:
            if hasattr(current, attr):
                current = getattr(current, attr)
            else:
                return None
        
        return current
    
    def _is_rope_module(self, module: nn.Module) -> bool:
        """Check if a module is a RoPE implementation."""
        if module is None:
            return False
        
        # Check class name for RoPE indicators - be more specific
        class_name = module.__class__.__name__.lower()
        rope_indicators = ['rotarypositionembedding', 'rotaryembedding', 'ropembedding', 'rope']
        
        # Require exact match for class name indicators
        has_rope_class_name = any(indicator in class_name for indicator in rope_indicators)
        
        # Check for RoPE-specific attributes - be more strict
        rope_attributes = ['inv_freq', 'cos_cached', 'sin_cached']
        has_rope_attributes = sum(hasattr(module, attr) for attr in rope_attributes) >= 2
        
        # Check for methods that suggest RoPE functionality
        rope_methods = ['forward', '__call__']
        has_methods = all(hasattr(module, method) for method in rope_methods)
        
        # Require either strong class name match OR (attributes AND methods)
        is_rope = has_rope_class_name or (has_rope_attributes and has_methods)
        
        # Additional validation: check if it actually has frequency-related tensors
        if is_rope and hasattr(module, 'inv_freq'):
            try:
                inv_freq = getattr(module, 'inv_freq')
                if not isinstance(inv_freq, torch.Tensor) or inv_freq.numel() == 0:
                    is_rope = False
            except Exception:
                is_rope = False
        
        return is_rope
    
    def _find_attention_modules(self, model: nn.Module) -> List[Tuple[str, nn.Module]]:
        """Find attention modules in the model."""
        attention_modules = []
        
        for name, module in model.named_modules():
            # Look for attention-related names
            name_lower = name.lower()
            if any(term in name_lower for term in ['attention', 'attn', 'self_attn', 'multi_head']):
                attention_modules.append((name, module))
        
        return attention_modules
    
    def _enhance_config_from_module(self, config: RoPEConfig, module: nn.Module) -> RoPEConfig:
        """Enhance RoPE config with information from the actual module."""
        # Try to get more accurate parameters from the module
        if hasattr(module, 'inv_freq'):
            try:
                inv_freq = module.inv_freq
                if isinstance(inv_freq, torch.Tensor) and inv_freq.numel() > 0:
                    # Calculate theta from inv_freq
                    # inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
                    # So theta = (1.0 / inv_freq[0]) ** (dim / 2)
                    first_inv_freq = inv_freq[0].item()
                    if first_inv_freq > 0:
                        calculated_theta = 1.0 / first_inv_freq
                        if calculated_theta != config.theta:
                            logger.debug(f"Updating theta from module: {calculated_theta} (was {config.theta})")
                            config.theta = calculated_theta
            except Exception as e:
                logger.debug(f"Could not extract theta from module: {e}")
        
        if hasattr(module, 'dim'):
            module_dim = getattr(module, 'dim')
            if isinstance(module_dim, int) and module_dim != config.head_dim:
                logger.debug(f"Updating head_dim from module: {module_dim} (was {config.head_dim})")
                config.head_dim = module_dim
        
        return config
    
    def _extract_config_from_module(self, module: nn.Module, fallback_config: Any) -> Optional[RoPEConfig]:
        """Extract RoPE config directly from module when config doesn't have RoPE info."""
        try:
            # Try to extract basic parameters from module
            theta = 10000.0  # Default
            head_dim = None
            max_pos = 2048  # Default
            
            if hasattr(module, 'inv_freq'):
                inv_freq = module.inv_freq
                if isinstance(inv_freq, torch.Tensor) and inv_freq.numel() > 0:
                    head_dim = inv_freq.numel() * 2  # inv_freq has half the dimensions
                    # Estimate theta from first element
                    first_inv_freq = inv_freq[0].item()
                    if first_inv_freq > 0:
                        theta = 1.0 / first_inv_freq
            
            if hasattr(module, 'dim'):
                head_dim = getattr(module, 'dim')
            
            # Use fallback config for missing values
            if head_dim is None and fallback_config:
                hidden_size = getattr(fallback_config, 'hidden_size', getattr(fallback_config, 'd_model', None))
                num_heads = getattr(fallback_config, 'num_attention_heads', getattr(fallback_config, 'num_heads', None))
                if hidden_size and num_heads:
                    head_dim = hidden_size // num_heads
            
            if head_dim is None:
                logger.warning("Could not determine head_dim from module")
                return None
            
            return RoPEConfig(
                theta=theta,
                head_dim=head_dim,
                max_position_embeddings=max_pos,
                scaling_factor=1.0,
                rope_type="default",
                partial_rotary_factor=1.0
            )
        
        except Exception as e:
            logger.debug(f"Could not extract config from module: {e}")
            return None
    
    def _check_cos_sin_cache(self, module: nn.Module) -> bool:
        """Check if module uses cos/sin caching."""
        if module is None:
            return False
        
        cache_attributes = ['cos_cached', 'sin_cached', '_cos_cached', '_sin_cached']
        return any(hasattr(module, attr) for attr in cache_attributes)
    
    def _check_dynamic_scaling(self, config: Any, module: nn.Module) -> bool:
        """Check if model supports dynamic RoPE scaling."""
        # Check config for scaling support
        if config and hasattr(config, 'rope_scaling'):
            scaling_info = getattr(config, 'rope_scaling')
            if isinstance(scaling_info, dict):
                return scaling_info.get('type') in ['linear', 'dynamic']
        
        # Check module for scaling methods
        if module:
            scaling_methods = ['_set_cos_sin_cache', 'update_cos_sin_cache', 'extend_cache']
            return any(hasattr(module, method) for method in scaling_methods)
        
        return False
    
    def generate_rope_embeddings(
        self, 
        rope_info: RoPEInfo, 
        seq_len: int, 
        device: torch.device, 
        dtype: torch.dtype = torch.float32
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate RoPE embeddings using detected configuration.
        
        Args:
            rope_info: Detected RoPE information
            seq_len: Sequence length
            device: Target device
            dtype: Target dtype
            
        Returns:
            Tuple of (cos, sin) tensors or None if no RoPE
        """
        if not rope_info.has_rope or not rope_info.config:
            return None
        
        # First, try to use model's own RoPE implementation
        if rope_info.rope_module:
            rope_result = self._use_model_rope(rope_info.rope_module, seq_len, device, dtype)
            if rope_result is not None:
                logger.debug("Used model's own RoPE implementation")
                return rope_result
        
        # Fallback to generating RoPE using detected config
        return self._generate_rope_from_config(rope_info.config, seq_len, device, dtype)
    
    def _use_model_rope(
        self, 
        rope_module: nn.Module, 
        seq_len: int, 
        device: torch.device, 
        dtype: torch.dtype
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Try to use the model's own RoPE implementation."""
        try:
            # Move module to correct device
            rope_module = rope_module.to(device)
            
            # Try different ways to call the RoPE module
            positions = torch.arange(seq_len, device=device, dtype=torch.long)
            
            # Method 1: Call with positions and seq_len
            if hasattr(rope_module, 'forward'):
                try:
                    result = rope_module.forward(positions, seq_len=seq_len)
                    if isinstance(result, tuple) and len(result) == 2:
                        cos, sin = result
                        if isinstance(cos, torch.Tensor) and isinstance(sin, torch.Tensor):
                            return cos.to(dtype), sin.to(dtype)
                except Exception as e:
                    logger.debug(f"RoPE forward method failed: {e}")
            
            # Method 2: Direct call
            if callable(rope_module):
                try:
                    result = rope_module(positions, seq_len=seq_len)
                    if isinstance(result, tuple) and len(result) == 2:
                        cos, sin = result
                        if isinstance(cos, torch.Tensor) and isinstance(sin, torch.Tensor):
                            return cos.to(dtype), sin.to(dtype)
                except Exception as e:
                    logger.debug(f"RoPE direct call failed: {e}")
            
            # Method 3: Try with just positions
            try:
                result = rope_module(positions)
                if isinstance(result, tuple) and len(result) == 2:
                    cos, sin = result
                    if isinstance(cos, torch.Tensor) and isinstance(sin, torch.Tensor):
                        return cos.to(dtype), sin.to(dtype)
            except Exception as e:
                logger.debug(f"RoPE positions-only call failed: {e}")
            
        except Exception as e:
            logger.debug(f"Could not use model's RoPE implementation: {e}")
        
        return None
    
    def _generate_rope_from_config(
        self, 
        config: RoPEConfig, 
        seq_len: int, 
        device: torch.device, 
        dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate RoPE embeddings from configuration."""
        logger.debug(f"Generating RoPE from config: theta={config.theta}, head_dim={config.head_dim}")
        
        # Create position indices
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        
        # Calculate dimension range (only use even dimensions for RoPE)
        dim_range = torch.arange(0, config.head_dim, 2, device=device, dtype=torch.float32)
        
        # Calculate inverse frequencies
        inv_freq = 1.0 / (config.theta ** (dim_range / config.head_dim))
        
        # Apply scaling if configured
        if config.scaling_factor != 1.0:
            if config.rope_type == "linear":
                inv_freq = inv_freq / config.scaling_factor
            elif config.rope_type == "dynamic":
                # Dynamic scaling (more complex, simplified here)
                scale = max(1.0, seq_len / config.max_position_embeddings)
                inv_freq = inv_freq / scale
        
        # Create frequency matrix
        freqs = torch.outer(positions, inv_freq)  # [seq_len, head_dim//2]
        
        # Generate cos and sin
        cos = freqs.cos().to(dtype)  # [seq_len, head_dim//2]
        sin = freqs.sin().to(dtype)  # [seq_len, head_dim//2]
        
        # Expand dimensions for broadcasting: [1, seq_len, head_dim//2]
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        
        logger.debug(f"Generated RoPE: cos.shape={cos.shape}, sin.shape={sin.shape}")
        return cos, sin
    
    def clear_cache(self):
        """Clear the RoPE cache."""
        self.rope_cache.clear()
        logger.info("Cleared RoPE handler cache")


# Global instance
universal_rope_handler = UniversalRoPEHandler()