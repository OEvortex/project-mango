"""
Universal Parameter Detector for Transformer Models

This module provides dynamic parameter detection and signature analysis for any
transformer model, eliminating the need for hardcoded parameter mappings.
Uses pure introspection and transformers conventions.
"""

import torch
import torch.nn as nn
import inspect
from typing import Dict, List, Set, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ParameterCategory(Enum):
    """Categories of parameters in transformer models."""
    INPUT = "input"                    # Primary input (hidden_states, inputs_embeds, input_ids)
    ATTENTION = "attention"            # Attention masks and related
    POSITION = "position"             # Position IDs, embeddings, cache positions
    CACHING = "caching"               # Past key values, use_cache
    OUTPUT_CONTROL = "output_control" # output_attentions, return_dict, etc.
    ENCODER_DECODER = "encoder_decoder" # Cross-attention, encoder states
    VISION = "vision"                 # Pixel values, image features
    AUDIO = "audio"                   # Audio/speech inputs
    SPECIALIZED = "specialized"       # Model-specific parameters
    TRAINING = "training"             # Labels, loss-related
    TIME_SERIES = "time_series"       # Time series specific
    MULTIMODAL = "multimodal"         # Multi-modal inputs


@dataclass
class ParameterInfo:
    """Information about a model parameter."""
    name: str
    category: ParameterCategory
    required: bool
    has_default: bool
    default_value: Any
    annotation: Optional[type]
    description: str


@dataclass
class ModelSignature:
    """Complete signature information for a model's forward method."""
    model_name: str
    parameters: Dict[str, ParameterInfo]
    required_params: Set[str]
    optional_params: Set[str]
    parameter_categories: Dict[ParameterCategory, Set[str]]
    supports_variable_kwargs: bool
    total_param_count: int


class UniversalParameterDetector:
    """
    Universal parameter detector for transformer models.
    
    Uses pure introspection to analyze model forward signatures and categorize
    parameters without any hardcoded mappings. Works with any transformer
    architecture supported by the transformers library.
    """
    
    # Parameter categorization patterns - these are discovery patterns, not hardcoded mappings
    PARAMETER_PATTERNS = {
        ParameterCategory.INPUT: {
            'names': ['hidden_states', 'inputs_embeds', 'input_ids', 'input_values', 'inputs'],
            'keywords': ['input', 'embed', 'hidden']
        },
        ParameterCategory.ATTENTION: {
            'names': ['attention_mask', 'head_mask', 'cross_attn_head_mask', 'encoder_attention_mask'],
            'keywords': ['attention', 'mask', 'attn']
        },
        ParameterCategory.POSITION: {
            'names': ['position_ids', 'position_embeddings', 'cache_position', 'position_bias'],
            'keywords': ['position', 'pos', 'cache']
        },
        ParameterCategory.CACHING: {
            'names': ['past_key_values', 'past_key_value', 'use_cache', 'cache'],
            'keywords': ['past', 'cache', 'key_value']
        },
        ParameterCategory.OUTPUT_CONTROL: {
            'names': ['output_attentions', 'output_hidden_states', 'return_dict'],
            'keywords': ['output', 'return']
        },
        ParameterCategory.ENCODER_DECODER: {
            'names': ['encoder_hidden_states', 'encoder_outputs', 'decoder_input_ids', 
                     'decoder_attention_mask', 'decoder_inputs_embeds'],
            'keywords': ['encoder', 'decoder', 'cross']
        },
        ParameterCategory.VISION: {
            'names': ['pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw'],
            'keywords': ['pixel', 'image', 'video', 'vision']
        },
        ParameterCategory.AUDIO: {
            'names': ['input_features', 'input_values', 'audio_features'],
            'keywords': ['audio', 'speech', 'features']
        },
        ParameterCategory.TRAINING: {
            'names': ['labels', 'mc_labels', 'mc_token_ids', 'logits_to_keep'],
            'keywords': ['label', 'loss', 'target']
        },
        ParameterCategory.SPECIALIZED: {
            'names': ['token_type_ids', 'langs', 'lengths', 'rope_deltas'],
            'keywords': ['type', 'lang', 'special']
        },
        ParameterCategory.TIME_SERIES: {
            'names': ['past_values', 'past_time_features', 'future_values', 'future_time_features'],
            'keywords': ['time', 'series', 'temporal']
        },
        ParameterCategory.MULTIMODAL: {
            'names': ['text_inputs', 'vision_inputs', 'audio_inputs'],
            'keywords': ['text', 'modal', 'multi']
        }
    }
    
    def __init__(self):
        """Initialize the universal parameter detector."""
        self.signature_cache: Dict[str, ModelSignature] = {}
    
    def analyze_model_signature(self, model: nn.Module, model_name: str = None) -> ModelSignature:
        """
        Analyze a model's forward method signature dynamically.
        
        Args:
            model: The transformer model to analyze
            model_name: Optional name for caching
            
        Returns:
            ModelSignature object with complete parameter information
        """
        if model_name and model_name in self.signature_cache:
            return self.signature_cache[model_name]
        
        if model_name is None:
            model_name = model.__class__.__name__
        
        logger.info(f"Analyzing signature for {model_name}")
        
        # Get the forward method
        forward_method = getattr(model, 'forward', None)
        if forward_method is None:
            raise ValueError(f"Model {model_name} has no forward method")
        
        # Inspect the signature
        signature = inspect.signature(forward_method)
        
        # Analyze parameters
        parameters = {}
        required_params = set()
        optional_params = set()
        parameter_categories = {category: set() for category in ParameterCategory}
        supports_variable_kwargs = False
        
        for param_name, param in signature.parameters.items():
            # Skip 'self' parameter
            if param_name == 'self':
                continue
            
            # Check for **kwargs
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                supports_variable_kwargs = True
                continue
            
            # Determine if required
            is_required = param.default == inspect.Parameter.empty
            has_default = not is_required
            default_value = param.default if has_default else None
            
            # Categorize parameter
            category = self._categorize_parameter(param_name, param)
            
            # Create parameter info
            param_info = ParameterInfo(
                name=param_name,
                category=category,
                required=is_required,
                has_default=has_default,
                default_value=default_value,
                annotation=param.annotation if param.annotation != inspect.Parameter.empty else None,
                description=self._get_parameter_description(param_name, category)
            )
            
            parameters[param_name] = param_info
            parameter_categories[category].add(param_name)
            
            if is_required:
                required_params.add(param_name)
            else:
                optional_params.add(param_name)
        
        # Create signature object
        model_signature = ModelSignature(
            model_name=model_name,
            parameters=parameters,
            required_params=required_params,
            optional_params=optional_params,
            parameter_categories=parameter_categories,
            supports_variable_kwargs=supports_variable_kwargs,
            total_param_count=len(parameters)
        )
        
        # Cache result
        if model_name:
            self.signature_cache[model_name] = model_signature
        
        logger.info(f"Detected {len(parameters)} parameters for {model_name}")
        logger.debug(f"Categories: {[(cat.value, len(params)) for cat, params in parameter_categories.items() if params]}")
        
        return model_signature
    
    def _categorize_parameter(self, param_name: str, param: inspect.Parameter) -> ParameterCategory:
        """Categorize a parameter based on its name and characteristics."""
        param_name_lower = param_name.lower()
        
        # Direct name match first
        for category, patterns in self.PARAMETER_PATTERNS.items():
            if param_name in patterns['names']:
                return category
        
        # Keyword-based categorization
        for category, patterns in self.PARAMETER_PATTERNS.items():
            for keyword in patterns['keywords']:
                if keyword in param_name_lower:
                    return category
        
        # Special handling for common patterns
        if 'id' in param_name_lower and 'input' in param_name_lower:
            return ParameterCategory.INPUT
        elif 'mask' in param_name_lower:
            return ParameterCategory.ATTENTION
        elif 'embed' in param_name_lower:
            return ParameterCategory.INPUT
        elif 'output' in param_name_lower:
            return ParameterCategory.OUTPUT_CONTROL
        
        # Default to specialized if we can't categorize
        return ParameterCategory.SPECIALIZED
    
    def _get_parameter_description(self, param_name: str, category: ParameterCategory) -> str:
        """Generate a description for a parameter based on its name and category."""
        descriptions = {
            ParameterCategory.INPUT: f"Primary input parameter: {param_name}",
            ParameterCategory.ATTENTION: f"Attention-related parameter: {param_name}",
            ParameterCategory.POSITION: f"Position-related parameter: {param_name}",
            ParameterCategory.CACHING: f"Caching parameter: {param_name}",
            ParameterCategory.OUTPUT_CONTROL: f"Output control parameter: {param_name}",
            ParameterCategory.ENCODER_DECODER: f"Encoder-decoder parameter: {param_name}",
            ParameterCategory.VISION: f"Vision input parameter: {param_name}",
            ParameterCategory.AUDIO: f"Audio input parameter: {param_name}",
            ParameterCategory.TRAINING: f"Training parameter: {param_name}",
            ParameterCategory.SPECIALIZED: f"Specialized parameter: {param_name}",
            ParameterCategory.TIME_SERIES: f"Time series parameter: {param_name}",
            ParameterCategory.MULTIMODAL: f"Multimodal parameter: {param_name}"
        }
        return descriptions.get(category, f"Parameter: {param_name}")
    
    def get_compatible_parameters(
        self, 
        model_signature: ModelSignature, 
        available_inputs: Dict[str, Any],
        include_categories: Optional[List[ParameterCategory]] = None,
        exclude_categories: Optional[List[ParameterCategory]] = None
    ) -> Dict[str, Any]:
        """
        Get compatible parameters for a model based on available inputs.
        
        Args:
            model_signature: The model's signature information
            available_inputs: Available input data
            include_categories: Only include these parameter categories
            exclude_categories: Exclude these parameter categories
            
        Returns:
            Dictionary of compatible parameters
        """
        compatible = {}
        
        for param_name, param_info in model_signature.parameters.items():
            # Apply category filters
            if include_categories and param_info.category not in include_categories:
                continue
            if exclude_categories and param_info.category in exclude_categories:
                continue
            
            # Check if we have this input available
            if param_name in available_inputs:
                value = available_inputs[param_name]
                if value is not None:
                    compatible[param_name] = value
            elif not param_info.required and param_info.has_default:
                # Use default value for optional parameters
                if param_info.default_value is not None:
                    compatible[param_name] = param_info.default_value
        
        return compatible
    
    def generate_parameter_strategies(self, model_signature: ModelSignature) -> List[Dict[str, Any]]:
        """
        Generate progressive parameter strategies based on model signature.
        
        Args:
            model_signature: The model's signature information
            
        Returns:
            List of strategy configurations ordered from most comprehensive to minimal
        """
        strategies = []
        
        # Strategy 1: All available parameters
        strategies.append({
            'name': 'comprehensive',
            'description': 'All detected parameters',
            'include_categories': list(ParameterCategory),
            'exclude_categories': [],
            'require_primary_input': True
        })
        
        # Strategy 2: Core transformer parameters only
        core_categories = [
            ParameterCategory.INPUT,
            ParameterCategory.ATTENTION,
            ParameterCategory.POSITION,
            ParameterCategory.OUTPUT_CONTROL
        ]
        strategies.append({
            'name': 'core_transformer',
            'description': 'Core transformer parameters only',
            'include_categories': core_categories,
            'exclude_categories': [],
            'require_primary_input': True
        })
        
        # Strategy 3: Essential parameters only
        essential_categories = [
            ParameterCategory.INPUT,
            ParameterCategory.ATTENTION
        ]
        strategies.append({
            'name': 'essential',
            'description': 'Essential parameters only',
            'include_categories': essential_categories,
            'exclude_categories': [],
            'require_primary_input': True
        })
        
        # Strategy 4: Minimal - just primary input
        strategies.append({
            'name': 'minimal',
            'description': 'Primary input only',
            'include_categories': [ParameterCategory.INPUT],
            'exclude_categories': [],
            'require_primary_input': True
        })
        
        return strategies
    
    def get_primary_input_parameter(self, model_signature: ModelSignature) -> Optional[str]:
        """Get the primary input parameter name for a model."""
        input_params = model_signature.parameter_categories[ParameterCategory.INPUT]
        
        # Preference order for primary input
        preference_order = ['hidden_states', 'inputs_embeds', 'input_ids', 'input_values', 'inputs']
        
        for preferred in preference_order:
            if preferred in input_params:
                return preferred
        
        # Return first input parameter if none match preferences
        if input_params:
            return next(iter(input_params))
        
        return None
    
    def clear_cache(self):
        """Clear the signature cache."""
        self.signature_cache.clear()
        logger.info("Cleared parameter detector signature cache")


# Global instance
universal_parameter_detector = UniversalParameterDetector()