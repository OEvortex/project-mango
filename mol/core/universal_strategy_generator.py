"""
Universal Strategy Generator for Transformer Models

This module provides dynamic strategy generation for parameter passing to any
transformer model, eliminating hardcoded strategy definitions. Generates
progressive fallback strategies based on model signature analysis.
"""

import torch
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from .universal_parameter_detector import (
    UniversalParameterDetector, ModelSignature, ParameterCategory, 
    universal_parameter_detector
)
from .universal_rope_handler import UniversalRoPEHandler, RoPEInfo, universal_rope_handler

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Types of parameter passing strategies."""
    COMPREHENSIVE = "comprehensive"      # All available parameters
    CORE_TRANSFORMER = "core_transformer"  # Core transformer parameters
    ATTENTION_FOCUSED = "attention_focused"  # Focus on attention parameters
    ESSENTIAL = "essential"             # Essential parameters only
    MINIMAL = "minimal"                 # Minimal viable parameters
    INPUT_ONLY = "input_only"          # Just primary input
    IDENTITY = "identity"              # Return input unchanged


@dataclass
class ParameterStrategy:
    """A strategy for passing parameters to a model."""
    name: str
    strategy_type: StrategyType
    description: str
    include_categories: Set[ParameterCategory]
    exclude_categories: Set[ParameterCategory]
    require_primary_input: bool
    allow_defaults: bool
    filter_none_values: bool
    priority: int  # Lower number = higher priority
    
    def __hash__(self):
        return hash(self.name)


class UniversalStrategyGenerator:
    """
    Universal strategy generator for transformer models.
    
    Dynamically generates parameter passing strategies based on model signature
    analysis, eliminating the need for hardcoded strategy definitions.
    """
    
    def __init__(
        self, 
        parameter_detector: Optional[UniversalParameterDetector] = None,
        rope_handler: Optional[UniversalRoPEHandler] = None
    ):
        """Initialize strategy generator."""
        self.parameter_detector = parameter_detector or universal_parameter_detector
        self.rope_handler = rope_handler or universal_rope_handler
        self.strategy_cache: Dict[str, List[ParameterStrategy]] = {}
    
    def generate_strategies(
        self, 
        model_signature: ModelSignature,
        rope_info: Optional[RoPEInfo] = None
    ) -> List[ParameterStrategy]:
        """
        Generate progressive fallback strategies for a model.
        
        Args:
            model_signature: The model's signature information
            rope_info: Optional RoPE information
            
        Returns:
            List of strategies ordered by priority (most comprehensive first)
        """
        cache_key = f"{model_signature.model_name}_{hash(frozenset(model_signature.parameters.keys()))}"
        
        if cache_key in self.strategy_cache:
            return self.strategy_cache[cache_key]
        
        logger.info(f"Generating strategies for {model_signature.model_name}")
        
        strategies = []
        
        # Strategy 1: Comprehensive - all detected parameters
        if self._has_sufficient_parameters(model_signature):
            strategies.append(ParameterStrategy(
                name="comprehensive",
                strategy_type=StrategyType.COMPREHENSIVE,
                description="All detected model parameters",
                include_categories=set(ParameterCategory),
                exclude_categories=set(),
                require_primary_input=True,
                allow_defaults=True,
                filter_none_values=True,
                priority=1
            ))
        
        # Strategy 2: Core transformer - essential transformer components
        core_categories = self._get_core_categories(model_signature, rope_info)
        strategies.append(ParameterStrategy(
            name="core_transformer",
            strategy_type=StrategyType.CORE_TRANSFORMER,
            description="Core transformer parameters",
            include_categories=core_categories,
            exclude_categories={ParameterCategory.TRAINING, ParameterCategory.SPECIALIZED},
            require_primary_input=True,
            allow_defaults=True,
            filter_none_values=True,
            priority=2
        ))
        
        # Strategy 3: Attention-focused - for attention-heavy models
        if self._has_attention_parameters(model_signature):
            attention_categories = {
                ParameterCategory.INPUT,
                ParameterCategory.ATTENTION,
                ParameterCategory.OUTPUT_CONTROL
            }
            strategies.append(ParameterStrategy(
                name="attention_focused",
                strategy_type=StrategyType.ATTENTION_FOCUSED,
                description="Input and attention parameters",
                include_categories=attention_categories,
                exclude_categories={ParameterCategory.TRAINING, ParameterCategory.SPECIALIZED},
                require_primary_input=True,
                allow_defaults=False,
                filter_none_values=True,
                priority=3
            ))
        
        # Strategy 4: Essential - input and basic attention
        essential_categories = {ParameterCategory.INPUT, ParameterCategory.ATTENTION}
        strategies.append(ParameterStrategy(
            name="essential",
            strategy_type=StrategyType.ESSENTIAL,
            description="Essential input and attention parameters",
            include_categories=essential_categories,
            exclude_categories=set(),
            require_primary_input=True,
            allow_defaults=False,
            filter_none_values=True,
            priority=4
        ))
        
        # Strategy 5: Minimal - just primary input and attention mask
        strategies.append(ParameterStrategy(
            name="minimal",
            strategy_type=StrategyType.MINIMAL,
            description="Primary input with optional attention mask",
            include_categories={ParameterCategory.INPUT, ParameterCategory.ATTENTION},
            exclude_categories=set(),
            require_primary_input=True,
            allow_defaults=False,
            filter_none_values=True,
            priority=5
        ))
        
        # Strategy 6: Input only - just the primary input parameter
        strategies.append(ParameterStrategy(
            name="input_only",
            strategy_type=StrategyType.INPUT_ONLY,
            description="Primary input parameter only",
            include_categories={ParameterCategory.INPUT},
            exclude_categories=set(),
            require_primary_input=True,
            allow_defaults=False,
            filter_none_values=True,
            priority=6
        ))
        
        # Strategy 7: Identity fallback - return input unchanged
        strategies.append(ParameterStrategy(
            name="identity",
            strategy_type=StrategyType.IDENTITY,
            description="Identity fallback - no model call",
            include_categories=set(),
            exclude_categories=set(),
            require_primary_input=False,
            allow_defaults=False,
            filter_none_values=False,
            priority=999
        ))
        
        # Cache and return
        self.strategy_cache[cache_key] = strategies
        
        logger.info(f"Generated {len(strategies)} strategies for {model_signature.model_name}")
        return strategies
    
    def _has_sufficient_parameters(self, model_signature: ModelSignature) -> bool:
        """Check if model has sufficient parameters for comprehensive strategy."""
        # Need at least input and some other parameters
        has_input = bool(model_signature.parameter_categories[ParameterCategory.INPUT])
        total_params = sum(len(params) for params in model_signature.parameter_categories.values())
        return has_input and total_params >= 3
    
    def _get_core_categories(
        self, 
        model_signature: ModelSignature, 
        rope_info: Optional[RoPEInfo]
    ) -> Set[ParameterCategory]:
        """Determine core parameter categories for a model."""
        core_categories = {
            ParameterCategory.INPUT,
            ParameterCategory.ATTENTION,
            ParameterCategory.OUTPUT_CONTROL
        }
        
        # Add position if model uses RoPE or has position parameters
        if (rope_info and rope_info.has_rope) or model_signature.parameter_categories[ParameterCategory.POSITION]:
            core_categories.add(ParameterCategory.POSITION)
        
        # Add caching if model supports it
        if model_signature.parameter_categories[ParameterCategory.CACHING]:
            core_categories.add(ParameterCategory.CACHING)
        
        # Add encoder-decoder if it's that type of model
        if model_signature.parameter_categories[ParameterCategory.ENCODER_DECODER]:
            core_categories.add(ParameterCategory.ENCODER_DECODER)
        
        return core_categories
    
    def _has_attention_parameters(self, model_signature: ModelSignature) -> bool:
        """Check if model has attention-related parameters."""
        attention_params = model_signature.parameter_categories[ParameterCategory.ATTENTION]
        return len(attention_params) > 0
    
    def build_strategy_parameters(
        self,
        strategy: ParameterStrategy,
        model_signature: ModelSignature,
        available_inputs: Dict[str, Any],
        rope_info: Optional[RoPEInfo] = None,
        sequence_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> Dict[str, Any]:
        """
        Build actual parameters for a strategy.
        
        Args:
            strategy: The strategy to apply
            model_signature: Model signature information
            available_inputs: Available input data
            rope_info: Optional RoPE information
            sequence_length: Optional sequence length for generating defaults
            device: Optional device for generating defaults
            dtype: Optional dtype for generating defaults
            
        Returns:
            Dictionary of parameters to pass to model
        """
        if strategy.strategy_type == StrategyType.IDENTITY:
            return {}
        
        parameters = {}
        
        # Get primary input parameter
        primary_input_param = self.parameter_detector.get_primary_input_parameter(model_signature)
        if strategy.require_primary_input and primary_input_param not in available_inputs:
            logger.warning(f"Primary input {primary_input_param} not available for strategy {strategy.name}")
            return {}
        
        # Process each parameter in the model signature
        for param_name, param_info in model_signature.parameters.items():
            # Check category inclusion/exclusion
            if strategy.include_categories and param_info.category not in strategy.include_categories:
                continue
            if param_info.category in strategy.exclude_categories:
                continue
            
            # Try to get value from available inputs
            if param_name in available_inputs:
                value = available_inputs[param_name]
                if not strategy.filter_none_values or value is not None:
                    parameters[param_name] = value
                    continue
            
            # Use defaults if allowed
            if strategy.allow_defaults and param_info.has_default:
                if param_info.default_value is not None:
                    parameters[param_name] = param_info.default_value
                    continue
            
            # Generate intelligent defaults for certain parameter types
            default_value = self._generate_parameter_default(
                param_name, param_info, available_inputs, rope_info, 
                sequence_length, device, dtype
            )
            if default_value is not None:
                parameters[param_name] = default_value
        
        logger.debug(f"Strategy '{strategy.name}' built {len(parameters)} parameters: {list(parameters.keys())}")
        return parameters
    
    def _generate_parameter_default(
        self,
        param_name: str,
        param_info,
        available_inputs: Dict[str, Any],
        rope_info: Optional[RoPEInfo],
        sequence_length: Optional[int],
        device: Optional[torch.device],
        dtype: Optional[torch.dtype]
    ) -> Any:
        """Generate intelligent defaults for parameters."""
        if not all([sequence_length, device, dtype]):
            return None
        
        # Handle RoPE position embeddings
        if param_name == 'position_embeddings' and rope_info and rope_info.has_rope:
            try:
                rope_result = self.rope_handler.generate_rope_embeddings(
                    rope_info, sequence_length, device, dtype
                )
                if rope_result:
                    return rope_result
            except Exception as e:
                logger.debug(f"Failed to generate RoPE embeddings: {e}")
        
        # Handle position IDs
        if param_name == 'position_ids' and sequence_length:
            batch_size = self._infer_batch_size(available_inputs)
            if batch_size:
                return torch.arange(sequence_length, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        
        # Handle cache position
        if param_name == 'cache_position' and sequence_length:
            return torch.arange(sequence_length, device=device, dtype=torch.long)
        
        # Handle common boolean parameters
        boolean_defaults = {
            'use_cache': False,
            'output_attentions': False,
            'output_hidden_states': False,
            'return_dict': False
        }
        if param_name in boolean_defaults:
            return boolean_defaults[param_name]
        
        # Handle None defaults for optional parameters
        none_defaults = {
            'past_key_values', 'past_key_value', 'head_mask', 'cross_attn_head_mask',
            'encoder_attention_mask', 'encoder_hidden_states', 'encoder_outputs'
        }
        if param_name in none_defaults:
            return None
        
        return None
    
    def _infer_batch_size(self, available_inputs: Dict[str, Any]) -> Optional[int]:
        """Infer batch size from available inputs."""
        for key, value in available_inputs.items():
            if isinstance(value, torch.Tensor) and value.dim() >= 2:
                return value.shape[0]
        return None
    
    def clear_cache(self):
        """Clear strategy cache."""
        self.strategy_cache.clear()
        logger.info("Cleared strategy generator cache")


# Global instance
universal_strategy_generator = UniversalStrategyGenerator()