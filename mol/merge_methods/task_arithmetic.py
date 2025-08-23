"""
Task Arithmetic merge method.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import logging

from .base_merge import BaseMergeMethod, MergeConfig

logger = logging.getLogger(__name__)


class TaskArithmeticMerge(BaseMergeMethod):
    """
    Task Arithmetic merge method.
    
    Implements task vector arithmetic for combining specialized models.
    Based on "Editing Models with Task Arithmetic".
    """
    
    def __init__(self, config: MergeConfig):
        super().__init__(config)
        
        # Task arithmetic parameters
        self.weights = config.parameters.get('weights', {})  # Task weights
        self.normalize = config.parameters.get('normalize', False)
        self.scaling_factor = config.parameters.get('scaling_factor', 1.0)
        
    def merge(self, models: Dict[str, nn.Module]) -> nn.Module:
        """
        Merge models using task arithmetic.
        
        Args:
            models: Dictionary of models to merge
            
        Returns:
            Merged model
        """
        if len(models) < 2:
            raise ValueError("Task arithmetic merge requires at least 2 models")
        
        self.validate_models(models)
        
        # Use base model if specified, otherwise use first model
        base_model_name = self.config.base_model
        if base_model_name is None:
            base_model_name = list(models.keys())[0]
        
        if base_model_name not in models:
            raise ValueError(f"Base model {base_model_name} not found in provided models")
        
        base_model = models[base_model_name]
        task_models = {k: v for k, v in models.items() if k != base_model_name}
        
        logger.info(f"Task arithmetic merging {len(task_models)} models with base {base_model_name}")
        
        # Create merged model
        merged_model = type(base_model)(base_model.config)
        merged_model.load_state_dict(base_model.state_dict())
        
        # Compute task vectors
        task_vectors = self._compute_task_vectors(base_model, task_models)
        
        # Combine task vectors
        combined_task_vector = self._combine_task_vectors(task_vectors)
        
        # Apply combined task vector to base model
        self._apply_task_vector(merged_model, base_model, combined_task_vector)
        
        logger.info("Task arithmetic merge completed")
        return merged_model
    
    def _compute_task_vectors(
        self,
        base_model: nn.Module,
        task_models: Dict[str, nn.Module]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Compute task vectors (differences from base model)."""
        base_weights = self.get_model_weights(base_model)
        task_vectors = {}
        
        for model_name, model in task_models.items():
            model_weights = self.get_model_weights(model)
            task_vector = {}
            
            for param_name in base_weights.keys():
                if param_name in model_weights:
                    diff = model_weights[param_name] - base_weights[param_name]
                    task_vector[param_name] = diff
                else:
                    task_vector[param_name] = torch.zeros_like(base_weights[param_name])
            
            # Normalize task vector if requested
            if self.normalize:
                task_vector = self._normalize_task_vector(task_vector)
            
            task_vectors[model_name] = task_vector
            logger.info(f"Computed task vector for {model_name}")
        
        return task_vectors
    
    def _normalize_task_vector(self, task_vector: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Normalize task vector."""
        # Compute overall norm
        total_norm_sq = 0.0
        for param_tensor in task_vector.values():
            total_norm_sq += torch.sum(param_tensor ** 2).item()
        
        total_norm = (total_norm_sq ** 0.5)
        
        if total_norm > 1e-8:
            normalized_vector = {}
            for param_name, param_tensor in task_vector.items():
                normalized_vector[param_name] = param_tensor / total_norm
            return normalized_vector
        
        return task_vector
    
    def _combine_task_vectors(
        self, 
        task_vectors: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Combine multiple task vectors using weights."""
        if not task_vectors:
            return {}
        
        # Get parameter names from first task vector
        param_names = list(next(iter(task_vectors.values())).keys())
        combined_vector = {}
        
        for param_name in param_names:
            combined_param = None
            total_weight = 0.0
            
            for model_name, task_vector in task_vectors.items():
                if param_name not in task_vector:
                    continue
                
                # Get weight for this model
                weight = self.weights.get(model_name, 1.0)
                total_weight += weight
                
                if combined_param is None:
                    combined_param = weight * task_vector[param_name]
                else:
                    combined_param += weight * task_vector[param_name]
            
            if combined_param is not None:
                # Average by total weight
                if total_weight > 0:
                    combined_param = combined_param / total_weight
                
                # Apply scaling factor
                combined_param = combined_param * self.scaling_factor
                combined_vector[param_name] = combined_param
            else:
                # Fallback to zero vector
                reference_tensor = next(iter(task_vectors.values()))[param_name]
                combined_vector[param_name] = torch.zeros_like(reference_tensor)
        
        return combined_vector
    
    def _apply_task_vector(
        self,
        target_model: nn.Module,
        base_model: nn.Module,
        task_vector: Dict[str, torch.Tensor]
    ):
        """Apply the combined task vector to the target model."""
        base_weights = self.get_model_weights(base_model)
        
        with torch.no_grad():
            for name, param in target_model.named_parameters():
                if name in task_vector and name in base_weights:
                    # Apply task vector: new_weight = base_weight + task_vector
                    new_weight = base_weights[name] + task_vector[name]
                    param.copy_(new_weight)
                elif name in base_weights:
                    # No task vector for this parameter, use base weight
                    param.copy_(base_weights[name])
        
        logger.info("Applied combined task vector to target model")
    
    def add_task_vector(
        self,
        base_model: nn.Module,
        task_model: nn.Module,
        weight: float = 1.0
    ) -> nn.Module:
        """
        Add a single task vector to a base model.
        
        Args:
            base_model: Base model
            task_model: Task-specific model
            weight: Weight for the task vector
            
        Returns:
            Model with added task vector
        """
        # Create result model
        result_model = type(base_model)(base_model.config)
        result_model.load_state_dict(base_model.state_dict())
        
        # Compute task vector
        base_weights = self.get_model_weights(base_model)
        task_weights = self.get_model_weights(task_model)
        
        with torch.no_grad():
            for name, param in result_model.named_parameters():
                if name in base_weights and name in task_weights:
                    task_vector = task_weights[name] - base_weights[name]
                    new_weight = base_weights[name] + weight * task_vector
                    param.copy_(new_weight)
        
        return result_model
    
    def subtract_task_vector(
        self,
        model: nn.Module,
        task_model: nn.Module, 
        base_model: nn.Module,
        weight: float = 1.0
    ) -> nn.Module:
        """
        Subtract a task vector from a model.
        
        Args:
            model: Model to modify
            task_model: Task-specific model
            base_model: Base model
            weight: Weight for the task vector
            
        Returns:
            Model with subtracted task vector
        """
        # Create result model
        result_model = type(model)(model.config)
        result_model.load_state_dict(model.state_dict())
        
        # Compute task vector
        base_weights = self.get_model_weights(base_model)
        task_weights = self.get_model_weights(task_model)
        model_weights = self.get_model_weights(model)
        
        with torch.no_grad():
            for name, param in result_model.named_parameters():
                if name in base_weights and name in task_weights and name in model_weights:
                    task_vector = task_weights[name] - base_weights[name]
                    new_weight = model_weights[name] - weight * task_vector
                    param.copy_(new_weight)
        
        return result_model
    
    def interpolate_task_vectors(
        self,
        base_model: nn.Module,
        task_models: List[nn.Module],
        weights: List[float]
    ) -> nn.Module:
        """
        Interpolate between multiple task vectors.
        
        Args:
            base_model: Base model
            task_models: List of task-specific models
            weights: Interpolation weights (should sum to 1.0)
            
        Returns:
            Model with interpolated task vectors
        """
        if len(task_models) != len(weights):
            raise ValueError("Number of task models must match number of weights")
        
        if abs(sum(weights) - 1.0) > 1e-6:
            logger.warning(f"Weights sum to {sum(weights)}, not 1.0")
        
        # Create result model
        result_model = type(base_model)(base_model.config)
        result_model.load_state_dict(base_model.state_dict())
        
        # Compute and combine task vectors
        base_weights = self.get_model_weights(base_model)
        
        with torch.no_grad():
            for name, param in result_model.named_parameters():
                if name in base_weights:
                    combined_task_vector = torch.zeros_like(base_weights[name])
                    
                    for task_model, weight in zip(task_models, weights):
                        task_weights = self.get_model_weights(task_model)
                        if name in task_weights:
                            task_vector = task_weights[name] - base_weights[name]
                            combined_task_vector += weight * task_vector
                    
                    new_weight = base_weights[name] + combined_task_vector
                    param.copy_(new_weight)
        
        return result_model