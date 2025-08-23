"""
Base class for merge methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import torch
import torch.nn as nn
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MergeConfig:
    """Configuration for merging operations."""
    method: str
    models: List[str]
    parameters: Dict[str, Any]
    dtype: str = "float16"
    device: str = "cpu"
    output_path: str = "./merged_model"
    base_model: Optional[str] = None
    layer_range: Optional[List[int]] = None


@dataclass
class LayerSlice:
    """Definition of a layer slice for merging."""
    model: str
    layer_range: List[int]
    weight: float = 1.0


class BaseMergeMethod(ABC):
    """
    Base class for all merge methods.
    """
    
    def __init__(self, config: MergeConfig):
        self.config = config
        self.models = {}
        self.model_configs = {}
        
    @abstractmethod
    def merge(self, models: Dict[str, nn.Module]) -> nn.Module:
        """
        Merge the models according to the method.
        
        Args:
            models: Dictionary mapping model names to loaded models
            
        Returns:
            Merged model
        """
        pass
    
    def load_models(self) -> Dict[str, nn.Module]:
        """Load all models specified in the config."""
        from transformers import AutoModel, AutoConfig
        
        models = {}
        for model_name in self.config.models:
            try:
                logger.info(f"Loading model: {model_name}")
                
                # Load config first
                config = AutoConfig.from_pretrained(model_name)
                self.model_configs[model_name] = config
                
                # Load model
                model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=getattr(torch, self.config.dtype),
                    device_map=self.config.device if self.config.device != "cpu" else None,
                    low_cpu_mem_usage=True
                )
                
                models[model_name] = model
                logger.info(f"Successfully loaded {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise
        
        return models
    
    def get_model_weights(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Extract weights from a model."""
        return {name: param.clone() for name, param in model.named_parameters()}
    
    def set_model_weights(self, model: nn.Module, weights: Dict[str, torch.Tensor]):
        """Set weights in a model."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in weights:
                    param.copy_(weights[name])
    
    def validate_models(self, models: Dict[str, nn.Module]) -> bool:
        """Validate that models can be merged."""
        if len(models) < 2:
            logger.error("Need at least 2 models to merge")
            return False
        
        # Check architecture compatibility
        first_model = next(iter(models.values()))
        first_config = next(iter(self.model_configs.values()))
        
        for model_name, model in models.items():
            config = self.model_configs[model_name]
            
            # Check hidden dimensions
            if config.hidden_size != first_config.hidden_size:
                logger.warning(
                    f"Model {model_name} has different hidden size: "
                    f"{config.hidden_size} vs {first_config.hidden_size}"
                )
            
            # Check number of layers
            if config.num_hidden_layers != first_config.num_hidden_layers:
                logger.warning(
                    f"Model {model_name} has different number of layers: "
                    f"{config.num_hidden_layers} vs {first_config.num_hidden_layers}"
                )
        
        return True
    
    def get_layers_in_range(self, model: nn.Module, layer_range: Optional[List[int]] = None) -> List[nn.Module]:
        """Get layers within specified range."""
        from ..core.block_extractor import BlockExtractor
        
        extractor = BlockExtractor()
        architecture_type = extractor.get_architecture_type(self.config.base_model or self.config.models[0])
        layers = extractor.get_model_layers(model, architecture_type)
        
        if layer_range is None:
            return list(layers)
        
        start, end = layer_range
        return list(layers[start:end])
    
    def interpolate_weights(
        self,
        weight1: torch.Tensor,
        weight2: torch.Tensor,
        t: float
    ) -> torch.Tensor:
        """Linear interpolation between two weight tensors."""
        return (1 - t) * weight1 + t * weight2
    
    def normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize tensor to unit norm."""
        norm = torch.norm(tensor, dim=-1, keepdim=True)
        return tensor / (norm + 1e-8)
    
    def save_merged_model(self, merged_model: nn.Module):
        """Save the merged model."""
        from transformers import AutoConfig
        
        # Use base model config as template
        base_config = AutoConfig.from_pretrained(
            self.config.base_model or self.config.models[0]
        )
        
        # Save config
        base_config.save_pretrained(self.config.output_path)
        
        # Save model
        merged_model.save_pretrained(self.config.output_path)
        
        logger.info(f"Saved merged model to {self.config.output_path}")