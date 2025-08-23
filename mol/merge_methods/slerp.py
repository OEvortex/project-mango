"""
Spherical Linear Interpolation (SLERP) merge method.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
import math
import logging

from .base_merge import BaseMergeMethod, MergeConfig

logger = logging.getLogger(__name__)


class SlerpMerge(BaseMergeMethod):
    """
    Spherical Linear Interpolation (SLERP) merge method.
    
    SLERP preserves the geometric properties of weight vectors during interpolation,
    making it superior to linear interpolation for high-dimensional spaces.
    """
    
    def __init__(self, config: MergeConfig):
        super().__init__(config)
        
        # SLERP parameters
        self.t = config.parameters.get('t', 0.5)  # Interpolation factor
        self.eps = config.parameters.get('eps', 1e-8)  # Numerical stability
        self.fallback_linear = config.parameters.get('fallback_linear', True)
        
        # Layer-specific parameters
        self.layer_params = config.parameters.get('layer_params', {})
        
    def merge(self, models: Dict[str, nn.Module]) -> nn.Module:
        """
        Merge two models using SLERP.
        
        Args:
            models: Dictionary with exactly 2 models to merge
            
        Returns:
            Merged model
        """
        if len(models) != 2:
            raise ValueError("SLERP merge requires exactly 2 models")
        
        self.validate_models(models)
        
        model_names = list(models.keys())
        model1, model2 = models[model_names[0]], models[model_names[1]]
        
        logger.info(f"Merging {model_names[0]} and {model_names[1]} using SLERP")
        
        # Create merged model (clone first model)
        merged_model = type(model1)(model1.config)
        merged_model.load_state_dict(model1.state_dict())
        
        # Get weights
        weights1 = self.get_model_weights(model1)
        weights2 = self.get_model_weights(model2)
        
        # Merge weights using SLERP
        merged_weights = self._slerp_merge_weights(weights1, weights2)
        
        # Set merged weights
        self.set_model_weights(merged_model, merged_weights)
        
        logger.info("SLERP merge completed")
        return merged_model
    
    def _slerp_merge_weights(
        self, 
        weights1: Dict[str, torch.Tensor], 
        weights2: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Merge weights using SLERP."""
        merged_weights = {}
        
        for name in weights1.keys():
            if name not in weights2:
                logger.warning(f"Parameter {name} not found in second model, using first model's weight")
                merged_weights[name] = weights1[name]
                continue
            
            w1 = weights1[name]
            w2 = weights2[name]
            
            if w1.shape != w2.shape:
                logger.warning(f"Shape mismatch for {name}: {w1.shape} vs {w2.shape}, using first model's weight")
                merged_weights[name] = w1
                continue
            
            # Get interpolation factor for this layer
            t = self._get_layer_t(name)
            
            # Perform SLERP
            merged_weights[name] = self._slerp_tensors(w1, w2, t)
        
        return merged_weights
    
    def _get_layer_t(self, layer_name: str) -> float:
        """Get interpolation factor for a specific layer."""
        # Check for layer-specific parameters
        for pattern, params in self.layer_params.items():
            if pattern in layer_name:
                if isinstance(params, dict) and 't' in params:
                    t_val = params['t']
                    if isinstance(t_val, list):
                        # Use different t values for different layer types
                        if 'self_attn' in layer_name and len(t_val) > 0:
                            return t_val[0]
                        elif 'mlp' in layer_name and len(t_val) > 1:
                            return t_val[1]
                        else:
                            return t_val[0] if t_val else self.t
                    return t_val
                return params
        
        return self.t
    
    def _slerp_tensors(self, t1: torch.Tensor, t2: torch.Tensor, t: float) -> torch.Tensor:
        """
        Perform SLERP between two tensors.
        
        Args:
            t1: First tensor
            t2: Second tensor  
            t: Interpolation factor [0, 1]
            
        Returns:
            Interpolated tensor
        """
        if t == 0.0:
            return t1.clone()
        if t == 1.0:
            return t2.clone()
        
        # Flatten tensors for computation
        original_shape = t1.shape
        v1 = t1.flatten()
        v2 = t2.flatten()
        
        # Normalize vectors
        v1_norm = torch.norm(v1)
        v2_norm = torch.norm(v2)
        
        if v1_norm < self.eps or v2_norm < self.eps:
            # One vector is near zero, use linear interpolation
            if self.fallback_linear:
                result = self.interpolate_weights(t1, t2, t)
                return result
            else:
                return t1.clone()
        
        v1_unit = v1 / v1_norm
        v2_unit = v2 / v2_norm
        
        # Compute dot product
        dot = torch.dot(v1_unit, v2_unit)
        dot = torch.clamp(dot, -1.0 + self.eps, 1.0 - self.eps)  # Numerical stability
        
        # Check if vectors are nearly collinear
        if abs(dot) > (1.0 - self.eps):
            # Vectors are nearly parallel/antiparallel, use linear interpolation
            if self.fallback_linear:
                result = self.interpolate_weights(t1, t2, t)
                return result
            else:
                return t1.clone()
        
        # Compute angle between vectors
        theta = math.acos(float(dot))
        sin_theta = math.sin(theta)
        
        if sin_theta < self.eps:
            # Angle is too small, use linear interpolation
            if self.fallback_linear:
                result = self.interpolate_weights(t1, t2, t)
                return result
            else:
                return t1.clone()
        
        # SLERP computation
        factor1 = math.sin((1.0 - t) * theta) / sin_theta
        factor2 = math.sin(t * theta) / sin_theta
        
        # Interpolate unit vectors
        v_slerp = factor1 * v1_unit + factor2 * v2_unit
        
        # Scale by interpolated norm
        interpolated_norm = (1.0 - t) * v1_norm + t * v2_norm
        result = v_slerp * interpolated_norm
        
        # Reshape back to original shape
        return result.view(original_shape)
    
    def gradient_slerp(
        self, 
        models: Dict[str, nn.Module], 
        layer_gradients: List[float]
    ) -> nn.Module:
        """
        SLERP with gradient-based interpolation factors.
        
        Args:
            models: Dictionary with exactly 2 models
            layer_gradients: List of t values for each layer
            
        Returns:
            Merged model with gradient-based interpolation
        """
        if len(models) != 2:
            raise ValueError("Gradient SLERP requires exactly 2 models")
        
        model_names = list(models.keys())
        model1, model2 = models[model_names[0]], models[model_names[1]]
        
        # Create merged model
        merged_model = type(model1)(model1.config)
        merged_model.load_state_dict(model1.state_dict())
        
        # Get layer weights
        from ..core.block_extractor import BlockExtractor
        extractor = BlockExtractor()
        layers1 = extractor.get_model_layers(model1, extractor.get_architecture_type(model_names[0]))
        layers2 = extractor.get_model_layers(model2, extractor.get_architecture_type(model_names[1]))
        merged_layers = extractor.get_model_layers(merged_model, extractor.get_architecture_type(model_names[0]))
        
        # Apply gradient-based SLERP to each layer
        for i, (layer1, layer2, merged_layer) in enumerate(zip(layers1, layers2, merged_layers)):
            if i < len(layer_gradients):
                t = layer_gradients[i]
                
                # Get layer weights
                weights1 = {name: param for name, param in layer1.named_parameters()}
                weights2 = {name: param for name, param in layer2.named_parameters()}
                
                # SLERP merge
                merged_weights = {}
                for name in weights1.keys():
                    if name in weights2:
                        merged_weights[name] = self._slerp_tensors(weights1[name], weights2[name], t)
                    else:
                        merged_weights[name] = weights1[name]
                
                # Set merged weights
                with torch.no_grad():
                    for name, param in merged_layer.named_parameters():
                        if name in merged_weights:
                            param.copy_(merged_weights[name])
        
        logger.info("Gradient SLERP merge completed")
        return merged_model