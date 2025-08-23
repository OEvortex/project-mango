"""
Block extraction utilities for different transformer architectures.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from transformers import (
    AutoModel, AutoConfig, AutoTokenizer,
    AutoModelForCausalLM, AutoModelForMaskedLM
)
from dataclasses import dataclass
import logging

from .universal_architecture import UniversalArchitectureHandler, ArchitectureInfo

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a model's architecture (legacy compatibility)."""
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
    
    Supports ALL transformer architectures available in HuggingFace (120+)
    with dynamic architecture detection and secure loading by default.
    """
    
    def __init__(self, trust_remote_code: bool = False):
        """
        Initialize BlockExtractor.
        
        Args:
            trust_remote_code: Whether to allow remote code execution (default: False for security)
        """
        self.trust_remote_code = trust_remote_code
        self.architecture_handler = UniversalArchitectureHandler(trust_remote_code)
        self.loaded_models: Dict[str, Any] = {}
        self.model_infos: Dict[str, ModelInfo] = {}
        self.architecture_infos: Dict[str, ArchitectureInfo] = {}
        
        if trust_remote_code:
            logger.warning(
                "⚠️  trust_remote_code=True enabled. This may execute arbitrary code from model repositories."
            )
    
    def get_architecture_info(self, model_name: str) -> ArchitectureInfo:
        """Get comprehensive architecture information for a model."""
        if model_name not in self.architecture_infos:
            arch_info = self.architecture_handler.detect_architecture(model_name)
            self.architecture_infos[model_name] = arch_info
        return self.architecture_infos[model_name]
    
    def get_architecture_type(self, model_name: str) -> str:
        """Determine the architecture type from model name or config (legacy compatibility)."""
        arch_info = self.get_architecture_info(model_name)
        return arch_info.architecture_type
    
    def load_model(self, model_name: str, device: str = "cpu") -> Tuple[nn.Module, ModelInfo]:
        """Load a model and extract its information."""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name], self.model_infos[model_name]
        
        try:
            # Get architecture information first
            arch_info = self.get_architecture_info(model_name)
            
            # Check if model requires remote code and we don't allow it
            if arch_info.requires_remote_code and not self.trust_remote_code:
                raise ValueError(
                    f"Model {model_name} requires trust_remote_code=True but it's disabled for security. "
                    "Set trust_remote_code=True if you trust this model repository."
                )
            
            # Load config and model
            config = AutoConfig.from_pretrained(
                model_name, 
                trust_remote_code=self.trust_remote_code
            )
            
            # Determine appropriate torch dtype
            torch_dtype = torch.float16 if device != "cpu" else torch.float32
            
            model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device if device != "cpu" else None,
                low_cpu_mem_usage=True,
                trust_remote_code=self.trust_remote_code
            )
            
            # Move to device if necessary
            if device == "cpu":
                model = model.to(device)
            
            # Create legacy ModelInfo for compatibility
            model_info = ModelInfo(
                model_name=model_name,
                hidden_dim=arch_info.hidden_dim,
                num_layers=arch_info.num_layers,
                num_attention_heads=arch_info.num_attention_heads,
                intermediate_size=arch_info.intermediate_size,
                vocab_size=arch_info.vocab_size,
                max_position_embeddings=arch_info.max_position_embeddings,
                layer_norm_eps=arch_info.layer_norm_eps,
                architecture_type=arch_info.architecture_type
            )
            
            # Cache the loaded model and info
            self.loaded_models[model_name] = model
            self.model_infos[model_name] = model_info
            
            logger.info(f"Loaded model {model_name}: {model_info}")
            return model, model_info
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            # Clear any partial cache entries
            self.loaded_models.pop(model_name, None)
            self.model_infos.pop(model_name, None)
            self.architecture_infos.pop(model_name, None)
            raise
    
    def get_model_layers(self, model: nn.Module, architecture_type: str) -> nn.ModuleList:
        """Get the transformer layers from a model."""
        # Get architecture info for the model
        arch_info = None
        for model_name, cached_info in self.architecture_infos.items():
            if cached_info.architecture_type == architecture_type:
                arch_info = cached_info
                break
        
        if arch_info:
            # Use universal handler with known architecture info
            return self.architecture_handler.get_layers(model, arch_info)
        else:
            # Fallback to legacy method for backward compatibility
            return self._get_layers_legacy(model, architecture_type)
    
    def _get_layers_legacy(self, model: nn.Module, architecture_type: str) -> nn.ModuleList:
        """Legacy layer detection method (for backward compatibility)."""
        # Legacy mapping for known architectures
        legacy_mapping = {
            "gpt2": "transformer.h",
            "gpt_neo": "transformer.h", 
            "gptj": "transformer.h",
            "llama": "model.layers",
            "bert": "encoder.layer",
            "roberta": "encoder.layer",
            "distilbert": "transformer.layer",
        }
        
        layer_attr = legacy_mapping.get(architecture_type)
        if not layer_attr:
            raise ValueError(f"Unsupported architecture type: {architecture_type}")
        
        # Navigate through nested attributes
        layers = model
        try:
            for attr in layer_attr.split('.'):
                if not hasattr(layers, attr):
                    raise AttributeError(f"Model does not have attribute '{attr}' in path '{layer_attr}'")
                layers = getattr(layers, attr)
            
            # Validate that we got a ModuleList or similar
            if not hasattr(layers, '__iter__') or not hasattr(layers, '__len__'):
                raise ValueError(f"Expected layers to be iterable, got {type(layers)}")
            
            return layers
            
        except AttributeError as e:
            logger.error(f"Failed to extract layers from {architecture_type} model: {e}")
            raise ValueError(f"Invalid layer path '{layer_attr}' for architecture '{architecture_type}': {e}")
    
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
        arch_info = self.get_architecture_info(model_name)
        
        try:
            # Use universal handler to get embeddings
            embeddings = self.architecture_handler.get_embeddings(model, arch_info)
            embed_dim = arch_info.hidden_dim
            return embeddings, embed_dim
            
        except Exception as e:
            # Fallback to legacy method
            logger.warning(f"Universal handler failed, trying legacy method: {e}")
            return self._get_embedding_layer_legacy(model, model_info)
    
    def _get_embedding_layer_legacy(self, model: nn.Module, model_info: ModelInfo) -> Tuple[nn.Module, int]:
        """Legacy embedding detection method."""
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
        # Check if we already have this model's LM head cached
        cache_key = f"{model_name}_lm_head"
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]
            
        try:
            arch_info = self.get_architecture_info(model_name)
            
            # Check if model requires remote code and we don't allow it
            if arch_info.requires_remote_code and not self.trust_remote_code:
                logger.warning(
                    f"Model {model_name} requires trust_remote_code=True for LM head, skipping"
                )
                return None
            
            # Load the full model with LM head
            from transformers import AutoModelForCausalLM, AutoModelForMaskedLM
            
            lm_head = None
            torch_dtype = torch.float16 if device != "cpu" else torch.float32
            
            # Try causal LM first if supported
            if arch_info.supports_causal_lm:
                try:
                    full_model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch_dtype,
                        device_map=device if device != "cpu" else None,
                        low_cpu_mem_usage=True,
                        trust_remote_code=self.trust_remote_code
                    )
                    
                    # Use universal handler to get LM head
                    if arch_info.lm_head_path:
                        lm_head = self.architecture_handler.get_lm_head(full_model, arch_info)
                    else:
                        lm_head = getattr(full_model, 'lm_head', None)
                    
                    if device == "cpu" and lm_head:
                        lm_head = lm_head.to(device)
                        
                except Exception as e1:
                    logger.debug(f"Could not load as causal LM: {e1}")
            
            # Try masked LM if causal LM failed and it's supported
            if not lm_head and arch_info.supports_masked_lm:
                try:
                    full_model = AutoModelForMaskedLM.from_pretrained(
                        model_name,
                        torch_dtype=torch_dtype,
                        device_map=device if device != "cpu" else None,
                        low_cpu_mem_usage=True,
                        trust_remote_code=self.trust_remote_code
                    )
                    
                    # Use universal handler to get LM head
                    if arch_info.lm_head_path:
                        lm_head = self.architecture_handler.get_lm_head(full_model, arch_info)
                    else:
                        lm_head = getattr(full_model, 'cls', None)
                        if not lm_head:
                            lm_head = getattr(full_model, 'lm_head', None)
                    
                    if device == "cpu" and lm_head:
                        lm_head = lm_head.to(device)
                        
                except Exception as e2:
                    logger.warning(f"Could not load LM head for {model_name}: {e2}")
                    lm_head = None
            
            # Cache the result (even if None)
            self.loaded_models[cache_key] = lm_head
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
        self.architecture_infos.clear()
        self.architecture_handler.clear_cache()
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Cleared all caches and freed memory")