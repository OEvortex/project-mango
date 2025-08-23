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
    trust_remote_code: bool = False  # Security: disable remote code by default


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
        
        # Security warning if trust_remote_code is enabled
        if config.trust_remote_code:
            logger.warning(
                "⚠️  trust_remote_code=True enabled. This may execute arbitrary code from model repositories. "
                "Only use with trusted models."
            )
        
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
        failed_models = []
        
        for model_name in self.config.models:
            try:
                logger.info(f"Loading model: {model_name}")
                
                # Load config first
                config = AutoConfig.from_pretrained(
                    model_name, 
                    trust_remote_code=self.config.trust_remote_code
                )
                self.model_configs[model_name] = config
                
                # Determine appropriate torch dtype
                if hasattr(torch, self.config.dtype):
                    torch_dtype = getattr(torch, self.config.dtype)
                else:
                    logger.warning(f"Unknown dtype {self.config.dtype}, using float16")
                    torch_dtype = torch.float16
                
                # Load model with appropriate settings
                model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    device_map=self.config.device if self.config.device != "cpu" else None,
                    low_cpu_mem_usage=True,
                    trust_remote_code=self.config.trust_remote_code
                )
                
                # Move to device if CPU
                if self.config.device == "cpu":
                    model = model.to(self.config.device)
                
                models[model_name] = model
                logger.info(f"Successfully loaded {model_name} ({sum(p.numel() for p in model.parameters()):,} parameters)")
                
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                failed_models.append(model_name)
                # Don't raise immediately, try to load other models
        
        if failed_models:
            if len(failed_models) == len(self.config.models):
                raise RuntimeError(f"Failed to load all models: {failed_models}")
            else:
                logger.warning(f"Failed to load some models: {failed_models}")
        
        if len(models) < 2:
            raise ValueError(f"Need at least 2 models to merge, only loaded {len(models)}")
        
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
        model_names = list(models.keys())
        first_model_name = model_names[0]
        first_config = self.model_configs[first_model_name]
        
        compatibility_issues = []
        
        for model_name in model_names[1:]:
            config = self.model_configs[model_name]
            
            # Check hidden dimensions
            if config.hidden_size != first_config.hidden_size:
                issue = f"Hidden size mismatch: {model_name}({config.hidden_size}) vs {first_model_name}({first_config.hidden_size})"
                compatibility_issues.append(issue)
                logger.warning(issue)
            
            # Check number of layers
            if config.num_hidden_layers != first_config.num_hidden_layers:
                issue = f"Layer count mismatch: {model_name}({config.num_hidden_layers}) vs {first_model_name}({first_config.num_hidden_layers})"
                compatibility_issues.append(issue)
                logger.warning(issue)
            
            # Check vocabulary size
            if config.vocab_size != first_config.vocab_size:
                issue = f"Vocab size mismatch: {model_name}({config.vocab_size}) vs {first_model_name}({first_config.vocab_size})"
                compatibility_issues.append(issue)
                logger.warning(issue)
            
            # Check attention heads
            if hasattr(config, 'num_attention_heads') and hasattr(first_config, 'num_attention_heads'):
                if config.num_attention_heads != first_config.num_attention_heads:
                    issue = f"Attention heads mismatch: {model_name}({config.num_attention_heads}) vs {first_model_name}({first_config.num_attention_heads})"
                    compatibility_issues.append(issue)
                    logger.warning(issue)
        
        # For strict merge methods, fail if there are critical incompatibilities
        if compatibility_issues and self.config.method in ['linear', 'slerp']:
            logger.error(f"Critical compatibility issues for {self.config.method} merge: {compatibility_issues}")
            return False
        
        if compatibility_issues:
            logger.warning(f"Found {len(compatibility_issues)} compatibility issues, but merge may still work with {self.config.method}")
        else:
            logger.info("All models are compatible")
        
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
        import os
        from transformers import AutoConfig
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(self.config.output_path, exist_ok=True)
            
            # Use base model config as template
            base_model_name = self.config.base_model or self.config.models[0]
            base_config = AutoConfig.from_pretrained(
                base_model_name, 
                trust_remote_code=self.config.trust_remote_code
            )
            
            # Add merge metadata to config
            base_config.mol_merge_info = {
                'method': self.config.method,
                'base_model': base_model_name,
                'merged_models': self.config.models,
                'merge_parameters': self.config.parameters,
                'merge_timestamp': str(torch.utils.data.get_worker_info() or 'unknown')
            }
            
            # Save config
            config_path = os.path.join(self.config.output_path, 'config.json')
            base_config.save_pretrained(self.config.output_path)
            logger.info(f"Saved config to {config_path}")
            
            # Save model weights
            model_path = os.path.join(self.config.output_path, 'pytorch_model.bin')
            torch.save(merged_model.state_dict(), model_path)
            logger.info(f"Saved model weights to {model_path}")
            
            # Try to save in HuggingFace format if possible
            try:
                merged_model.save_pretrained(self.config.output_path)
                logger.info(f"Saved model in HuggingFace format to {self.config.output_path}")
            except Exception as e:
                logger.warning(f"Could not save in HuggingFace format: {e}")
            
            # Save merge summary
            summary_path = os.path.join(self.config.output_path, 'merge_summary.txt')
            with open(summary_path, 'w') as f:
                f.write(f"MoL Merge Summary\n")
                f.write(f"================\n\n")
                f.write(f"Merge Method: {self.config.method}\n")
                f.write(f"Base Model: {base_model_name}\n")
                f.write(f"Merged Models: {', '.join(self.config.models)}\n")
                f.write(f"Output Path: {self.config.output_path}\n")
                f.write(f"Parameters: {self.config.parameters}\n")
                f.write(f"Device: {self.config.device}\n")
                f.write(f"Data Type: {self.config.dtype}\n")
            logger.info(f"Saved merge summary to {summary_path}")
            
            logger.info(f"Successfully saved merged model to {self.config.output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save merged model: {e}")
            raise