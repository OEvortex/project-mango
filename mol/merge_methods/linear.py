"""
Linear interpolation merge method.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import logging

from .base_merge import BaseMergeMethod, MergeConfig

logger = logging.getLogger(__name__)


class LinearMerge(BaseMergeMethod):
    """
    Linear interpolation merge method.
    
    Simple weighted average of model parameters.
    """
    
    def __init__(self, config: MergeConfig):
        super().__init__(config)
        
        # Linear merge parameters
        self.weights = config.parameters.get('weights', {})  # Model weights
        self.normalize_weights = config.parameters.get('normalize_weights', True)
        
    def merge(self, models: Dict[str, nn.Module]) -> nn.Module:
        """
        Merge models using linear interpolation.
        
        Args:
            models: Dictionary of models to merge
            
        Returns:
            Merged model
        """
        if len(models) < 2:
            raise ValueError("Linear merge requires at least 2 models")
        
        self.validate_models(models)
        
        model_names = list(models.keys())
        base_model = models[model_names[0]]
        
        logger.info(f"Linear merging {len(models)} models")
        
        # Create merged model
        merged_model = type(base_model)(base_model.config)
        merged_model.load_state_dict(base_model.state_dict())
        
        # Get model weights and merge them
        model_weights = {name: self.get_model_weights(model) for name, model in models.items()}
        merged_weights = self._linear_merge_weights(model_weights)
        
        # Set merged weights
        self.set_model_weights(merged_model, merged_weights)
        
        logger.info("Linear merge completed")
        return merged_model
    
    def _linear_merge_weights(self, model_weights: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Merge weights using linear interpolation."""
        if not model_weights:
            return {}
        
        # Get parameter names from first model
        param_names = list(next(iter(model_weights.values())).keys())
        merged_weights = {}
        
        # Normalize weights if requested
        merge_weights = self._get_normalized_weights(list(model_weights.keys()))
        
        for param_name in param_names:
            merged_param = None
            
            for model_name, weights in model_weights.items():
                if param_name not in weights:
                    continue
                
                weight = merge_weights.get(model_name, 1.0 / len(model_weights))
                
                if merged_param is None:
                    merged_param = weight * weights[param_name]
                else:
                    merged_param += weight * weights[param_name]
            
            if merged_param is not None:
                merged_weights[param_name] = merged_param
            else:
                # Fallback to first model's weight
                reference_weights = next(iter(model_weights.values()))
                merged_weights[param_name] = reference_weights[param_name]
        
        return merged_weights
    
    def _get_normalized_weights(self, model_names: List[str]) -> Dict[str, float]:
        """Get normalized merge weights."""
        weights = {}
        total_weight = 0.0
        
        # Get weights for each model
        for name in model_names:
            weight = self.weights.get(name, 1.0)
            weights[name] = weight
            total_weight += weight
        
        # Normalize if requested
        if self.normalize_weights and total_weight > 0:
            for name in weights:
                weights[name] /= total_weight
        
        return weights
    
    def merge_two_models(
        self,
        model1: nn.Module,
        model2: nn.Module, 
        alpha: float = 0.5
    ) -> nn.Module:
        """
        Merge exactly two models with a single interpolation parameter.
        
        Args:
            model1: First model
            model2: Second model
            alpha: Interpolation factor [0, 1]. 0 = pure model1, 1 = pure model2
            
        Returns:
            Merged model
        """
        # Create result model
        merged_model = type(model1)(model1.config)
        merged_model.load_state_dict(model1.state_dict())
        
        weights1 = self.get_model_weights(model1)
        weights2 = self.get_model_weights(model2)
        
        # Linear interpolation
        with torch.no_grad():
            for name, param in merged_model.named_parameters():
                if name in weights1 and name in weights2:
                    interpolated = (1 - alpha) * weights1[name] + alpha * weights2[name]
                    param.copy_(interpolated)
                elif name in weights1:
                    param.copy_(weights1[name])
        
        return merged_model
    
    def layer_wise_merge(
        self,
        models: Dict[str, nn.Module],
        layer_weights: Dict[int, Dict[str, float]]
    ) -> nn.Module:
        """
        Merge models with different weights for different layers.
        
        Args:
            models: Dictionary of models to merge
            layer_weights: Dict mapping layer index to model weights for that layer
            
        Returns:
            Merged model with layer-wise weights
        """
        model_names = list(models.keys())
        base_model = models[model_names[0]]
        
        # Create merged model
        merged_model = type(base_model)(base_model.config)
        merged_model.load_state_dict(base_model.state_dict())
        
        # Get layers from each model
        from ..core.block_extractor import BlockExtractor
        extractor = BlockExtractor()
        
        model_layers = {}
        for name, model in models.items():
            arch_type = extractor.get_architecture_type(name)
            layers = extractor.get_model_layers(model, arch_type)
            model_layers[name] = layers
        
        # Get merged model layers
        arch_type = extractor.get_architecture_type(model_names[0])
        merged_layers = extractor.get_model_layers(merged_model, arch_type)
        
        # Merge each layer with its specific weights
        for layer_idx in range(len(merged_layers)):
            if layer_idx in layer_weights:
                weights = layer_weights[layer_idx]
                
                # Collect layer weights from all models
                layer_model_weights = {}
                for model_name in model_names:
                    if model_name in model_layers and layer_idx < len(model_layers[model_name]):
                        layer_weights_dict = {
                            name: param for name, param in model_layers[model_name][layer_idx].named_parameters()
                        }
                        layer_model_weights[model_name] = layer_weights_dict
                
                # Merge this layer
                merged_layer_weights = self._linear_merge_weights_with_specific_weights(
                    layer_model_weights, weights
                )
                
                # Set merged weights to the layer
                with torch.no_grad():
                    for name, param in merged_layers[layer_idx].named_parameters():
                        if name in merged_layer_weights:
                            param.copy_(merged_layer_weights[name])
        
        return merged_model
    
    def _linear_merge_weights_with_specific_weights(
        self,
        model_weights: Dict[str, Dict[str, torch.Tensor]],
        specific_weights: Dict[str, float]
    ) -> Dict[str, torch.Tensor]:
        """Merge weights with specific weight values."""
        if not model_weights:
            return {}
        
        param_names = list(next(iter(model_weights.values())).keys())
        merged_weights = {}
        
        # Normalize specific weights
        total_weight = sum(specific_weights.values())
        if total_weight > 0:
            normalized_weights = {k: v / total_weight for k, v in specific_weights.items()}
        else:
            normalized_weights = {k: 1.0 / len(specific_weights) for k in specific_weights}
        
        for param_name in param_names:
            merged_param = None
            
            for model_name, weights in model_weights.items():
                if param_name not in weights or model_name not in normalized_weights:
                    continue
                
                weight = normalized_weights[model_name]
                
                if merged_param is None:
                    merged_param = weight * weights[param_name]
                else:
                    merged_param += weight * weights[param_name]
            
            if merged_param is not None:
                merged_weights[param_name] = merged_param
        
        return merged_weights