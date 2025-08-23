"""
TIES (Trim, Elect Sign, Disjoint Merge) method.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging

from .base_merge import BaseMergeMethod, MergeConfig

logger = logging.getLogger(__name__)


class TiesMerge(BaseMergeMethod):
    """
    TIES-Merging: Resolving Interference when Merging Models.
    
    Implements the TIES algorithm that:
    1. Trims redundant parameters
    2. Elects unified signs
    3. Disjoint merges parameters
    """
    
    def __init__(self, config: MergeConfig):
        super().__init__(config)
        
        # TIES parameters
        self.density = config.parameters.get('density', 0.8)  # Fraction of parameters to keep
        self.normalize = config.parameters.get('normalize', True)
        self.majority_sign_method = config.parameters.get('majority_sign_method', 'total')
        
    def merge(self, models: Dict[str, nn.Module]) -> nn.Module:
        """
        Merge multiple models using TIES.
        
        Args:
            models: Dictionary of models to merge
            
        Returns:
            Merged model
        """
        if len(models) < 2:
            raise ValueError("TIES merge requires at least 2 models")
        
        self.validate_models(models)
        
        # Use base model if specified, otherwise use first model
        base_model_name = self.config.base_model
        if base_model_name is None:
            base_model_name = list(models.keys())[0]
        
        if base_model_name not in models:
            raise ValueError(f"Base model {base_model_name} not found in provided models")
        
        base_model = models[base_model_name]
        task_models = {k: v for k, v in models.items() if k != base_model_name}
        
        logger.info(f"TIES merging {len(task_models)} models with base {base_model_name}")
        
        # Create merged model
        merged_model = type(base_model)(base_model.config)
        merged_model.load_state_dict(base_model.state_dict())
        
        # Get task vectors (differences from base model)
        task_vectors = self._compute_task_vectors(base_model, task_models)
        
        # TIES merge process
        merged_task_vector = self._ties_merge(task_vectors)
        
        # Apply merged task vector to base model
        self._apply_task_vector(merged_model, base_model, merged_task_vector)
        
        logger.info("TIES merge completed")
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
                    task_vector[param_name] = model_weights[param_name] - base_weights[param_name]
                else:
                    task_vector[param_name] = torch.zeros_like(base_weights[param_name])
            
            task_vectors[model_name] = task_vector
            logger.info(f"Computed task vector for {model_name}")
        
        return task_vectors
    
    def _ties_merge(self, task_vectors: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Apply TIES algorithm to merge task vectors."""
        if not task_vectors:
            return {}
        
        # Get parameter names from first task vector
        param_names = list(next(iter(task_vectors.values())).keys())
        merged_vector = {}
        
        for param_name in param_names:
            # Collect all task vectors for this parameter
            param_vectors = []
            for model_name, task_vector in task_vectors.items():
                if param_name in task_vector:
                    param_vectors.append(task_vector[param_name])
            
            if not param_vectors:
                merged_vector[param_name] = torch.zeros_like(
                    next(iter(task_vectors.values()))[param_name]
                )
                continue
            
            # Apply TIES algorithm to this parameter
            merged_vector[param_name] = self._ties_merge_parameter(param_vectors, param_name)
        
        return merged_vector
    
    def _ties_merge_parameter(
        self, 
        param_vectors: List[torch.Tensor], 
        param_name: str
    ) -> torch.Tensor:
        """Apply TIES algorithm to a single parameter."""
        if len(param_vectors) == 1:
            return param_vectors[0]
        
        # Stack all vectors
        stacked = torch.stack(param_vectors)  # [num_models, ...param_shape]
        
        # Step 1: Trim (keep top-k% of parameters by magnitude)
        trimmed_vectors = self._trim_parameters(stacked)
        
        # Step 2: Elect Sign (resolve sign conflicts)
        sign_vector = self._elect_sign(trimmed_vectors)
        
        # Step 3: Disjoint Merge (average parameters with consistent signs)
        merged = self._disjoint_merge(trimmed_vectors, sign_vector)
        
        return merged
    
    def _trim_parameters(self, stacked_vectors: torch.Tensor) -> torch.Tensor:
        """Trim parameters by keeping only top-k% by magnitude."""
        # Flatten each vector for easier processing
        original_shape = stacked_vectors.shape
        num_models = original_shape[0]
        flattened = stacked_vectors.view(num_models, -1)
        
        # For each model, keep only top density% of parameters
        trimmed = torch.zeros_like(flattened)
        
        for i in range(num_models):
            vector = flattened[i]
            abs_vector = torch.abs(vector)
            
            # Find threshold for top density% of parameters
            k = int(self.density * vector.numel())
            if k > 0:
                threshold = torch.topk(abs_vector, k)[0][-1]
                mask = abs_vector >= threshold
                trimmed[i] = vector * mask
        
        return trimmed.view(original_shape)
    
    def _elect_sign(self, trimmed_vectors: torch.Tensor) -> torch.Tensor:
        """Elect unified sign vector based on majority voting."""
        # Get signs of non-zero elements
        signs = torch.sign(trimmed_vectors)  # [num_models, ...param_shape]
        
        if self.majority_sign_method == 'total':
            # Sum signs across models
            sign_sum = torch.sum(signs, dim=0)
            unified_sign = torch.sign(sign_sum)
        elif self.majority_sign_method == 'mass':
            # Weight signs by magnitude
            magnitudes = torch.abs(trimmed_vectors)
            weighted_signs = signs * magnitudes
            sign_sum = torch.sum(weighted_signs, dim=0)
            unified_sign = torch.sign(sign_sum)
        else:
            raise ValueError(f"Unknown majority sign method: {self.majority_sign_method}")
        
        return unified_sign
    
    def _disjoint_merge(
        self, 
        trimmed_vectors: torch.Tensor, 
        sign_vector: torch.Tensor
    ) -> torch.Tensor:
        """Merge parameters that agree with the unified sign."""
        num_models = trimmed_vectors.shape[0]
        
        # Create masks for parameters that agree with unified sign
        signs = torch.sign(trimmed_vectors)
        agreement_mask = (signs == sign_vector.unsqueeze(0)) & (trimmed_vectors != 0)
        
        # Sum parameters that agree with unified sign
        agreed_sum = torch.sum(trimmed_vectors * agreement_mask.float(), dim=0)
        
        # Count how many models agree for each parameter
        agreement_count = torch.sum(agreement_mask.float(), dim=0)
        
        # Average among agreeing models (avoid division by zero)
        merged = torch.where(
            agreement_count > 0,
            agreed_sum / agreement_count,
            torch.zeros_like(agreed_sum)
        )
        
        # Normalize if requested
        if self.normalize:
            merged = self._normalize_merged_vector(merged)
        
        return merged
    
    def _normalize_merged_vector(self, merged_vector: torch.Tensor) -> torch.Tensor:
        """Normalize the merged vector."""
        # Simple L2 normalization per parameter tensor
        norm = torch.norm(merged_vector)
        if norm > 1e-8:
            return merged_vector / norm
        return merged_vector
    
    def _apply_task_vector(
        self,
        target_model: nn.Module,
        base_model: nn.Module, 
        task_vector: Dict[str, torch.Tensor]
    ):
        """Apply the merged task vector to the target model."""
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
        
        logger.info("Applied merged task vector to target model")