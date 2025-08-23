"""
HuggingFace model implementations for MoL system.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
from transformers import (
    AutoModel, AutoConfig, AutoTokenizer,
    AutoModelForCausalLM, AutoModelForMaskedLM
)
import logging

from .base_model import BaseMoLModel
from ..core.universal_architecture import UniversalArchitectureHandler, ArchitectureInfo

logger = logging.getLogger(__name__)


class HuggingFaceModel(BaseMoLModel):
    """
    HuggingFace model wrapper for MoL system.
    
    Supports ALL transformer architectures available through HuggingFace (120+)
    with dynamic architecture detection and secure loading by default.
    """
    
    def __init__(self, model_name: str, model_type: str = "base", trust_remote_code: bool = False):
        """
        Initialize HuggingFace model.
        
        Args:
            model_name: HuggingFace model identifier
            model_type: Type of model to load ("base", "causal_lm", "masked_lm")
            trust_remote_code: Whether to allow remote code execution (default: False for security)
        """
        super().__init__(model_name)
        self.model_type = model_type
        self.trust_remote_code = trust_remote_code
        self.architecture_handler = UniversalArchitectureHandler(trust_remote_code)
        self.architecture_info: Optional[ArchitectureInfo] = None
        
        if trust_remote_code:
            logger.warning(
                "⚠️  trust_remote_code=True enabled. This may execute arbitrary code from model repositories."
            )
    
    def load_model(self, device: str = "cpu", **kwargs) -> nn.Module:
        """Load the HuggingFace model."""
        try:
            # Get architecture information first
            self.architecture_info = self.architecture_handler.detect_architecture(self.model_name)
            
            # Check if model requires remote code and we don't allow it
            if self.architecture_info.requires_remote_code and not self.trust_remote_code:
                raise ValueError(
                    f"Model {self.model_name} requires trust_remote_code=True but it's disabled for security. "
                    "Set trust_remote_code=True if you trust this model repository."
                )
            
            # Load config and tokenizer
            self.config = AutoConfig.from_pretrained(
                self.model_name, 
                trust_remote_code=self.trust_remote_code
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Determine dtype
            torch_dtype = kwargs.get('torch_dtype', torch.float16)
            
            # Load model based on type and architecture capabilities
            if self.model_type == "causal_lm" and self.architecture_info.supports_causal_lm:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype,
                    device_map=device if device != "cpu" else None,
                    low_cpu_mem_usage=True,
                    trust_remote_code=self.trust_remote_code,
                    **kwargs
                )
            elif self.model_type == "masked_lm" and self.architecture_info.supports_masked_lm:
                self.model = AutoModelForMaskedLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype,
                    device_map=device if device != "cpu" else None,
                    low_cpu_mem_usage=True,
                    trust_remote_code=self.trust_remote_code,
                    **kwargs
                )
            else:  # base model or unsupported specialized type
                if self.model_type not in ["base"] and not (
                    (self.model_type == "causal_lm" and self.architecture_info.supports_causal_lm) or
                    (self.model_type == "masked_lm" and self.architecture_info.supports_masked_lm)
                ):
                    logger.warning(
                        f"Model type '{self.model_type}' not supported by {self.architecture_info.architecture_family} "
                        f"architecture, falling back to base model"
                    )
                
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype,
                    device_map=device if device != "cpu" else None,
                    low_cpu_mem_usage=True,
                    trust_remote_code=self.trust_remote_code,
                    **kwargs
                )
            
            # Move to CPU if needed
            if device == "cpu":
                self.model = self.model.to(device)
            
            logger.info(
                f"Loaded {self.model_name} as {self.model_type} model "
                f"({self.architecture_info.architecture_family})"
            )
            return self.model
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def get_layers(self) -> nn.ModuleList:
        """Get the transformer layers from the model."""
        if not self.model or not self.architecture_info:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        return self.architecture_handler.get_layers(self.model, self.architecture_info)
    
    def get_embeddings(self) -> nn.Module:
        """Get the embedding layer."""
        if not self.model or not self.architecture_info:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        return self.architecture_handler.get_embeddings(self.model, self.architecture_info)
    
    def get_lm_head(self) -> Optional[nn.Module]:
        """Get the language modeling head if available."""
        if not self.model or not self.architecture_info:
            return None
        
        return self.architecture_handler.get_lm_head(self.model, self.architecture_info)
    
    def get_hidden_dim(self) -> int:
        """Get the hidden dimension of the model."""
        if self.architecture_info:
            return self.architecture_info.hidden_dim
        if not self.config:
            raise RuntimeError("Model config not loaded. Call load_model() first.")
        return getattr(self.config, 'hidden_size', getattr(self.config, 'd_model', 768))
    
    def get_num_layers(self) -> int:
        """Get the number of transformer layers."""
        if self.architecture_info:
            return self.architecture_info.num_layers
        if not self.config:
            raise RuntimeError("Model config not loaded. Call load_model() first.")
        return getattr(self.config, 'num_hidden_layers', getattr(self.config, 'num_layers', 12))
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        if self.architecture_info:
            return self.architecture_info.vocab_size
        if not self.config:
            raise RuntimeError("Model config not loaded. Call load_model() first.")
        return self.config.vocab_size
    
    def get_attention_heads(self) -> int:
        """Get the number of attention heads."""
        if self.architecture_info:
            return self.architecture_info.num_attention_heads
        if not self.config:
            raise RuntimeError("Model config not loaded. Call load_model() first.")
        return getattr(self.config, 'num_attention_heads', getattr(self.config, 'num_heads', 12))
    
    def get_intermediate_size(self) -> int:
        """Get the intermediate size (FFN dimension)."""
        if self.architecture_info:
            return self.architecture_info.intermediate_size
        if not self.config:
            raise RuntimeError("Model config not loaded. Call load_model() first.")
        return getattr(self.config, 'intermediate_size', self.get_hidden_dim() * 4)
    
    def get_architecture_type(self) -> str:
        """Get the architecture type."""
        if self.architecture_info:
            return self.architecture_info.architecture_type
        return "unknown"
    
    def get_architecture_family(self) -> str:
        """Get the architecture family."""
        if self.architecture_info:
            return self.architecture_info.architecture_family
        return "UNKNOWN"
    
    def extract_layer(self, layer_idx: int) -> nn.Module:
        """Extract a specific transformer layer."""
        layers = self.get_layers()
        if layer_idx >= len(layers) or layer_idx < 0:
            raise ValueError(
                f"Layer index {layer_idx} out of range for model {self.model_name} "
                f"(has {len(layers)} layers)"
            )
        return layers[layer_idx]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        base_info = super().get_model_info()
        
        if self.config:
            base_info.update({
                "architecture_type": self.architecture_type,
                "num_attention_heads": self.get_attention_heads(),
                "intermediate_size": self.get_intermediate_size(),
                "max_position_embeddings": getattr(self.config, 'max_position_embeddings', None),
                "layer_norm_eps": getattr(self.config, 'layer_norm_eps', None),
                "model_type": self.model_type,
            })
        
        return base_info


def create_huggingface_model(
    model_name: str,
    model_type: str = "base",
    device: str = "cpu",
    **kwargs
) -> HuggingFaceModel:
    """
    Factory function to create and load a HuggingFace model.
    
    Args:
        model_name: HuggingFace model identifier
        model_type: Type of model to load ("base", "causal_lm", "masked_lm")
        device: Device to load the model on
        **kwargs: Additional arguments for model loading
    
    Returns:
        Loaded HuggingFaceModel instance
    """
    model = HuggingFaceModel(model_name, model_type)
    model.load_model(device, **kwargs)
    return model