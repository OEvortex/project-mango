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

logger = logging.getLogger(__name__)


class HuggingFaceModel(BaseMoLModel):
    """
    HuggingFace model wrapper for MoL system.
    
    Supports various transformer architectures available through HuggingFace.
    """
    
    # Mapping of architecture types to their layer attribute names
    LAYER_ATTR_MAPPING = {
        "gpt2": "transformer.h",
        "gpt_neo": "transformer.h", 
        "gptj": "transformer.h",
        "llama": "model.layers",
        "bert": "encoder.layer",
        "roberta": "encoder.layer",
        "distilbert": "transformer.layer",
    }
    
    def __init__(self, model_name: str, model_type: str = "base"):
        """
        Initialize HuggingFace model.
        
        Args:
            model_name: HuggingFace model identifier
            model_type: Type of model to load ("base", "causal_lm", "masked_lm")
        """
        super().__init__(model_name)
        self.model_type = model_type
        self.architecture_type = None
    
    def load_model(self, device: str = "cpu", **kwargs) -> nn.Module:
        """Load the HuggingFace model."""
        try:
            # Load config and determine architecture
            self.config = AutoConfig.from_pretrained(self.model_name)
            self.architecture_type = self._determine_architecture_type()
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model based on type
            if self.model_type == "causal_lm":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map=device if device != "cpu" else None,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            elif self.model_type == "masked_lm":
                self.model = AutoModelForMaskedLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map=device if device != "cpu" else None,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            else:  # base model
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map=device if device != "cpu" else None,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            
            logger.info(f"Loaded {self.model_name} as {self.model_type} model")
            return self.model
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def _determine_architecture_type(self) -> str:
        """Determine the architecture type from model config."""
        arch_type = self.config.architectures[0].lower() if self.config.architectures else ""
        
        # Map known architecture types
        if "gpt2" in arch_type:
            return "gpt2"
        elif "gptneo" in arch_type:
            return "gpt_neo"
        elif "gptj" in arch_type:
            return "gptj"
        elif "llama" in arch_type:
            return "llama"
        elif "bert" in arch_type and "distil" not in arch_type:
            return "bert"
        elif "roberta" in arch_type:
            return "roberta"
        elif "distilbert" in arch_type:
            return "distilbert"
        else:
            logger.warning(f"Unknown architecture type: {arch_type}, defaulting to gpt2")
            return "gpt2"
    
    def get_layers(self) -> nn.ModuleList:
        """Get the transformer layers from the model."""
        if not self.model or not self.architecture_type:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        layer_attr = self.LAYER_ATTR_MAPPING.get(self.architecture_type)
        if not layer_attr:
            raise ValueError(f"Unsupported architecture type: {self.architecture_type}")
        
        # Navigate through nested attributes
        layers = self.model
        for attr in layer_attr.split('.'):
            layers = getattr(layers, attr)
        
        return layers
    
    def get_embeddings(self) -> nn.Module:
        """Get the embedding layer."""
        if not self.model or not self.architecture_type:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Different architectures have different embedding structures
        if self.architecture_type in ["gpt2", "gpt_neo", "gptj"]:
            return self.model.transformer.wte  # Word token embeddings
        elif self.architecture_type == "llama":
            return self.model.model.embed_tokens
        elif self.architecture_type in ["bert", "roberta"]:
            return self.model.embeddings
        elif self.architecture_type == "distilbert":
            return self.model.embeddings
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture_type}")
    
    def get_lm_head(self) -> Optional[nn.Module]:
        """Get the language modeling head if available."""
        if not self.model:
            return None
        
        # Try different LM head attributes
        for attr_name in ["lm_head", "cls", "classifier"]:
            if hasattr(self.model, attr_name):
                return getattr(self.model, attr_name)
        
        return None
    
    def get_hidden_dim(self) -> int:
        """Get the hidden dimension of the model."""
        if not self.config:
            raise RuntimeError("Model config not loaded. Call load_model() first.")
        return self.config.hidden_size
    
    def get_num_layers(self) -> int:
        """Get the number of transformer layers."""
        if not self.config:
            raise RuntimeError("Model config not loaded. Call load_model() first.")
        return self.config.num_hidden_layers
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        if not self.config:
            raise RuntimeError("Model config not loaded. Call load_model() first.")
        return self.config.vocab_size
    
    def get_attention_heads(self) -> int:
        """Get the number of attention heads."""
        if not self.config:
            raise RuntimeError("Model config not loaded. Call load_model() first.")
        return self.config.num_attention_heads
    
    def get_intermediate_size(self) -> int:
        """Get the intermediate size (FFN dimension)."""
        if not self.config:
            raise RuntimeError("Model config not loaded. Call load_model() first.")
        return getattr(self.config, 'intermediate_size', self.config.hidden_size * 4)
    
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