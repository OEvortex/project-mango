"""
Block extraction utilities for different transformer architectures.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from transformers import (
    AutoModel, AutoConfig, AutoTokenizer,
    GPT2Model, GPTNeoModel, GPTJModel, LlamaModel, 
    BertModel, RobertaModel, DistilBertModel
)
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a model's architecture."""
    model_name: str
    hidden_dim: int
    num_layers: int
    num_attention_heads: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    layer_norm_eps: float
    architecture_type: str


@dataclass 
class ExtractedBlock:
    """A transformer block extracted from a model."""
    block: nn.Module
    layer_idx: int
    model_name: str
    input_dim: int
    output_dim: int
    attention_heads: int
    intermediate_size: int


class BlockExtractor:
    """
    Extracts transformer blocks from different model architectures.
    
    Supports various HuggingFace model types and provides unified interface
    for accessing individual transformer layers.
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
    
    def __init__(self):
        self.loaded_models: Dict[str, Any] = {}
        self.model_infos: Dict[str, ModelInfo] = {}
    
    def get_architecture_type(self, model_name: str) -> str:
        """Determine the architecture type from model name or config."""
        try:
            config = AutoConfig.from_pretrained(model_name)
            arch_type = config.architectures[0].lower() if config.architectures else ""
            
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
                
        except Exception as e:
            logger.warning(f"Could not determine architecture for {model_name}: {e}")
            return "gpt2"
    
    def load_model(self, model_name: str, device: str = "cpu") -> Tuple[nn.Module, ModelInfo]:
        """Load a model and extract its information."""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name], self.model_infos[model_name]
        
        try:
            # Load config and model
            config = AutoConfig.from_pretrained(model_name)
            model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device if device != "cpu" else None,
                low_cpu_mem_usage=True
            )
            
            # Extract model information
            architecture_type = self.get_architecture_type(model_name)
            
            model_info = ModelInfo(
                model_name=model_name,
                hidden_dim=config.hidden_size,
                num_layers=config.num_hidden_layers,
                num_attention_heads=config.num_attention_heads,
                intermediate_size=getattr(config, 'intermediate_size', config.hidden_size * 4),
                vocab_size=config.vocab_size,
                max_position_embeddings=getattr(config, 'max_position_embeddings', 2048),
                layer_norm_eps=getattr(config, 'layer_norm_eps', 1e-5),
                architecture_type=architecture_type
            )
            
            # Cache the loaded model and info
            self.loaded_models[model_name] = model
            self.model_infos[model_name] = model_info
            
            logger.info(f"Loaded model {model_name}: {model_info}")
            return model, model_info
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def get_model_layers(self, model: nn.Module, architecture_type: str) -> nn.ModuleList:
        """Get the transformer layers from a model."""
        layer_attr = self.LAYER_ATTR_MAPPING.get(architecture_type)
        if not layer_attr:
            raise ValueError(f"Unsupported architecture type: {architecture_type}")
        
        # Navigate through nested attributes
        layers = model
        for attr in layer_attr.split('.'):
            layers = getattr(layers, attr)
        
        return layers
    
    def extract_block(
        self, 
        model_name: str, 
        layer_idx: int, 
        device: str = "cpu"
    ) -> ExtractedBlock:
        """Extract a specific transformer block from a model."""
        model, model_info = self.load_model(model_name, device)
        
        # Get the layers
        layers = self.get_model_layers(model, model_info.architecture_type)
        
        if layer_idx >= len(layers) or layer_idx < 0:
            raise ValueError(
                f"Layer index {layer_idx} out of range for model {model_name} "
                f"(has {len(layers)} layers)"
            )
        
        # Extract the specific block
        block = layers[layer_idx]
        
        return ExtractedBlock(
            block=block,
            layer_idx=layer_idx,
            model_name=model_name,
            input_dim=model_info.hidden_dim,
            output_dim=model_info.hidden_dim,
            attention_heads=model_info.num_attention_heads,
            intermediate_size=model_info.intermediate_size
        )
    
    def extract_multiple_blocks(
        self, 
        model_name: str, 
        layer_indices: List[int],
        device: str = "cpu"
    ) -> List[ExtractedBlock]:
        """Extract multiple transformer blocks from a model."""
        blocks = []
        for layer_idx in layer_indices:
            block = self.extract_block(model_name, layer_idx, device)
            blocks.append(block)
        return blocks
    
    def get_embedding_layer(self, model_name: str, device: str = "cpu") -> Tuple[nn.Module, int]:
        """Get the embedding layer from a model."""
        model, model_info = self.load_model(model_name, device)
        
        # Different architectures have different embedding structures
        if model_info.architecture_type in ["gpt2", "gpt_neo", "gptj"]:
            embeddings = model.transformer.wte  # Word token embeddings
            embed_dim = model_info.hidden_dim
        elif model_info.architecture_type == "llama":
            embeddings = model.model.embed_tokens
            embed_dim = model_info.hidden_dim
        elif model_info.architecture_type in ["bert", "roberta"]:
            embeddings = model.embeddings
            embed_dim = model_info.hidden_dim
        elif model_info.architecture_type == "distilbert":
            embeddings = model.embeddings
            embed_dim = model_info.hidden_dim
        else:
            raise ValueError(f"Unsupported architecture: {model_info.architecture_type}")
        
        return embeddings, embed_dim
    
    def get_lm_head(self, model_name: str, device: str = "cpu") -> Optional[nn.Module]:
        """Get the language modeling head from a model."""
        try:
            # Load the full model with LM head
            from transformers import AutoModelForCausalLM, AutoModelForMaskedLM
            
            model_info = self.model_infos.get(model_name)
            if not model_info:
                _, model_info = self.load_model(model_name, device)
            
            # Try causal LM first, then masked LM
            try:
                full_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map=device if device != "cpu" else None,
                    low_cpu_mem_usage=True
                )
                lm_head = getattr(full_model, 'lm_head', None)
            except:
                try:
                    full_model = AutoModelForMaskedLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        device_map=device if device != "cpu" else None,
                        low_cpu_mem_usage=True
                    )
                    lm_head = getattr(full_model, 'cls', None)
                except:
                    logger.warning(f"Could not load LM head for {model_name}")
                    return None
            
            return lm_head
            
        except Exception as e:
            logger.warning(f"Failed to get LM head for {model_name}: {e}")
            return None
    
    def get_compatible_models(
        self, 
        reference_model: str,
        candidate_models: List[str],
        max_dim_ratio: float = 4.0
    ) -> List[Tuple[str, ModelInfo]]:
        """
        Find models compatible for fusion with a reference model.
        
        Args:
            reference_model: Name of the reference model
            candidate_models: List of candidate models to check
            max_dim_ratio: Maximum ratio of hidden dimensions allowed
        
        Returns:
            List of (model_name, model_info) tuples for compatible models
        """
        ref_model, ref_info = self.load_model(reference_model)
        compatible = []
        
        for candidate in candidate_models:
            try:
                _, candidate_info = self.load_model(candidate)
                
                # Check dimension compatibility
                dim_ratio = max(
                    candidate_info.hidden_dim / ref_info.hidden_dim,
                    ref_info.hidden_dim / candidate_info.hidden_dim
                )
                
                if dim_ratio <= max_dim_ratio:
                    compatible.append((candidate, candidate_info))
                    logger.info(
                        f"Model {candidate} is compatible with {reference_model} "
                        f"(dim ratio: {dim_ratio:.2f})"
                    )
                else:
                    logger.warning(
                        f"Model {candidate} has incompatible dimensions "
                        f"(dim ratio: {dim_ratio:.2f} > {max_dim_ratio})"
                    )
                    
            except Exception as e:
                logger.error(f"Could not check compatibility for {candidate}: {e}")
        
        return compatible
    
    def clear_cache(self):
        """Clear cached models to free memory."""
        self.loaded_models.clear()
        self.model_infos.clear()
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()