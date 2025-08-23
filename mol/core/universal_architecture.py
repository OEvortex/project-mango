"""
Universal Architecture Handler for MoL System

This module provides comprehensive support for all transformer architectures
available in HuggingFace Transformers, with dynamic architecture detection
and secure model loading (trust_remote_code=False by default).
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
from transformers import AutoModel, AutoConfig, AutoTokenizer
from dataclasses import dataclass
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class ArchitectureInfo:
    """Comprehensive information about a model architecture."""
    model_name: str
    architecture_family: str  # DECODER_ONLY, ENCODER_ONLY, ENCODER_DECODER, VISION, MULTIMODAL
    architecture_type: str    # specific type like "llama", "bert", "gpt2"
    layer_path: str          # path to transformer layers
    embedding_path: str      # path to embeddings
    lm_head_path: Optional[str]  # path to language modeling head
    hidden_dim: int
    num_layers: int
    num_attention_heads: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    layer_norm_eps: float
    supports_causal_lm: bool
    supports_masked_lm: bool
    requires_remote_code: bool


class UniversalArchitectureHandler:
    """
    Universal handler for all transformer architectures.
    
    Dynamically detects architecture types and component paths,
    supporting 120+ transformer architectures without hardcoded mappings.
    """
    
    # Common layer path patterns (ordered by likelihood)
    LAYER_PATH_PATTERNS = [
        # Decoder-only patterns (most common)
        "transformer.h",           # GPT-2, GPT-Neo, GPT-J, Falcon
        "model.layers",            # Llama, Mistral, Qwen, Yi, CodeLlama
        "transformer.blocks",      # MPT, Mosaic models
        "gpt_neox.layers",         # GPT-NeoX, Pythia
        "model.decoder.layers",    # OPT, BLOOM decoder
        "transformer.layers",      # Some custom models
        "model.transformer.layers", # Some wrapped models
        
        # Encoder-only patterns
        "encoder.layer",           # BERT, RoBERTa, ELECTRA, DeBERTa
        "encoder.layers",          # Some BERT variants
        "transformer.layer",       # DistilBERT
        "albert.encoder.albert_layer_groups", # ALBERT
        "roberta.encoder.layer",   # Some RoBERTa variants
        
        # Encoder-decoder patterns
        "encoder.block",           # T5 encoder, UL2
        "decoder.block",           # T5 decoder, UL2
        "model.encoder.layers",    # BART, Pegasus encoder
        "model.decoder.layers",    # BART, Pegasus decoder
        "encoder.layers",          # Generic encoder
        "decoder.layers",          # Generic decoder
        
        # Vision transformer patterns
        "encoder.layer",           # ViT, DeiT (same as BERT)
        "encoder.layers",          # Swin Transformer
        "layers",                  # Some vision models
        "blocks",                  # Some vision models
        "transformer.resblocks",   # CLIP vision encoder
        
        # Multimodal patterns
        "vision_model.encoder.layers",    # CLIP vision
        "text_model.encoder.layer",       # CLIP text
        "language_model.model.layers",    # Some multimodal
        
        # Less common patterns
        "h",                       # Direct layer access
        "block",                   # Single block models
        "transformer_blocks",      # Alternative naming
    ]
    
    # Embedding path patterns
    EMBEDDING_PATH_PATTERNS = [
        # Text embeddings
        "embeddings",                    # BERT, RoBERTa, DistilBERT
        "transformer.wte",               # GPT-2, GPT-Neo, GPT-J
        "model.embed_tokens",            # Llama, Mistral, OPT, T5
        "transformer.word_embeddings",   # GPT-NeoX
        "transformer.embedding",         # Some models
        "embed_tokens",                  # Direct access
        "word_embeddings",               # Direct access
        "shared",                        # T5 shared embeddings
        
        # Vision embeddings
        "embeddings.patch_embeddings",   # ViT
        "patch_embed",                   # Swin, DeiT variants
        "vision_model.embeddings",       # CLIP vision
        "embeddings.position_embeddings", # Some models
        
        # Multimodal embeddings
        "text_model.embeddings",         # CLIP text
        "language_model.model.embed_tokens", # Some multimodal
    ]
    
    # LM head path patterns
    LM_HEAD_PATH_PATTERNS = [
        "lm_head",                 # Most causal LMs
        "cls",                     # BERT MLM, classification
        "classifier",              # Classification models
        "prediction_head",         # Some models
        "head",                    # Generic head
        "output_projection",       # Some architectures
        "predictions",             # BERT variants
        "pooler",                  # BERT pooler
        "score",                   # Sequence classification
    ]
    
    def __init__(self, trust_remote_code: bool = False):
        """
        Initialize universal architecture handler.
        
        Args:
            trust_remote_code: Whether to allow remote code execution (default: False for security)
        """
        self.trust_remote_code = trust_remote_code
        self.architecture_cache: Dict[str, ArchitectureInfo] = {}
        
        if trust_remote_code:
            logger.warning(
                "⚠️  trust_remote_code=True enabled. This may execute arbitrary code from model repositories. "
                "Only use with trusted models."
            )
    
    def detect_architecture(self, model_name: str) -> ArchitectureInfo:
        """
        Detect architecture information for a model.
        
        Args:
            model_name: HuggingFace model identifier
            
        Returns:
            ArchitectureInfo object with detected information
        """
        # Check cache first
        if model_name in self.architecture_cache:
            return self.architecture_cache[model_name]
        
        logger.info(f"🔍 Detecting architecture for {model_name}")
        
        try:
            # Load model config
            config = AutoConfig.from_pretrained(
                model_name, 
                trust_remote_code=self.trust_remote_code
            )
            
            # Detect basic architecture info
            architecture_type = self._detect_architecture_type(config)
            architecture_family = self._classify_architecture_family(config, architecture_type)
            
            # Load model to introspect structure
            model = self._load_model_for_introspection(model_name, config)
            
            # Detect component paths
            layer_path = self._detect_layer_path(model, config)
            embedding_path = self._detect_embedding_path(model, config)
            lm_head_path = self._detect_lm_head_path(model, config)
            
            # Extract configuration details
            arch_info = ArchitectureInfo(
                model_name=model_name,
                architecture_family=architecture_family,
                architecture_type=architecture_type,
                layer_path=layer_path,
                embedding_path=embedding_path,
                lm_head_path=lm_head_path,
                hidden_dim=getattr(config, 'hidden_size', getattr(config, 'd_model', 768)),
                num_layers=getattr(config, 'num_hidden_layers', getattr(config, 'num_layers', 12)),
                num_attention_heads=getattr(config, 'num_attention_heads', getattr(config, 'num_heads', 12)),
                intermediate_size=getattr(config, 'intermediate_size', 
                                        getattr(config, 'hidden_size', 768) * 4),
                vocab_size=getattr(config, 'vocab_size', 50257),
                max_position_embeddings=getattr(config, 'max_position_embeddings', 
                                              getattr(config, 'max_seq_length', 2048)),
                layer_norm_eps=getattr(config, 'layer_norm_eps', 
                                     getattr(config, 'layernorm_epsilon', 1e-5)),
                supports_causal_lm=self._supports_causal_lm(config, architecture_family),
                supports_masked_lm=self._supports_masked_lm(config, architecture_family),
                requires_remote_code=self._requires_remote_code(model_name, config)
            )
            
            # Cache the result
            self.architecture_cache[model_name] = arch_info
            
            logger.info(
                f"✅ Detected {arch_info.architecture_type} ({arch_info.architecture_family}) "
                f"with {arch_info.num_layers} layers, hidden_dim={arch_info.hidden_dim}"
            )
            
            return arch_info
            
        except Exception as e:
            logger.error(f"❌ Failed to detect architecture for {model_name}: {e}")
            raise ValueError(f"Could not detect architecture for {model_name}: {e}")
    
    def _detect_architecture_type(self, config) -> str:
        """Detect specific architecture type from config."""
        # Try model_type first (most reliable)
        if hasattr(config, 'model_type') and config.model_type:
            return config.model_type.lower()
        
        # Try architectures list
        if hasattr(config, 'architectures') and config.architectures:
            arch_name = config.architectures[0]
            # Extract base name (e.g., "LlamaForCausalLM" -> "llama")
            base_name = re.sub(r'(For\w+|Model)$', '', arch_name).lower()
            return base_name
        
        # Fallback to config class name
        config_class = config.__class__.__name__
        base_name = re.sub(r'Config$', '', config_class).lower()
        return base_name
    
    def _classify_architecture_family(self, config, architecture_type: str) -> str:
        """Classify architecture into major family."""
        # Decoder-only models
        if any(name in architecture_type for name in [
            'gpt', 'llama', 'mistral', 'falcon', 'mpt', 'opt', 'bloom', 'qwen', 
            'yi', 'codellama', 'solar', 'phi', 'gemma', 'deepseek'
        ]):
            return "DECODER_ONLY"
        
        # Encoder-only models  
        if any(name in architecture_type for name in [
            'bert', 'roberta', 'electra', 'deberta', 'albert', 'distilbert'
        ]):
            return "ENCODER_ONLY"
        
        # Encoder-decoder models
        if any(name in architecture_type for name in [
            't5', 'bart', 'pegasus', 'marian', 'blenderbot', 'ul2'
        ]):
            return "ENCODER_DECODER"
        
        # Vision models
        if any(name in architecture_type for name in [
            'vit', 'deit', 'swin', 'beit', 'clip_vision', 'convnext'
        ]):
            return "VISION"
        
        # Multimodal models
        if any(name in architecture_type for name in [
            'clip', 'flava', 'layoutlm', 'lxmert', 'uniter'
        ]):
            return "MULTIMODAL"
        
        # Default to decoder-only for unknown types
        logger.warning(f"Unknown architecture type '{architecture_type}', assuming DECODER_ONLY")
        return "DECODER_ONLY"
    
    def _load_model_for_introspection(self, model_name: str, config) -> nn.Module:
        """Load model for architecture introspection."""
        try:
            return AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # Use float32 for introspection
                device_map="cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=self.trust_remote_code
            )
        except Exception as e:
            logger.warning(f"Could not load full model for introspection: {e}")
            # Return None, will try alternative detection methods
            return None
    
    def _detect_layer_path(self, model: Optional[nn.Module], config) -> str:
        """Detect path to transformer layers."""
        if model is None:
            # Fallback to pattern matching based on architecture
            architecture_type = self._detect_architecture_type(config)
            return self._get_fallback_layer_path(architecture_type)
        
        # Try each pattern
        for pattern in self.LAYER_PATH_PATTERNS:
            try:
                layers = self._navigate_to_attribute(model, pattern)
                if self._validate_transformer_layers(layers, config):
                    logger.debug(f"Found transformer layers at: {pattern}")
                    return pattern
            except (AttributeError, ValueError):
                continue
        
        # If no pattern works, try introspection
        layer_path = self._introspect_layer_path(model, config)
        if layer_path:
            return layer_path
        
        # Final fallback
        architecture_type = self._detect_architecture_type(config)
        fallback_path = self._get_fallback_layer_path(architecture_type)
        logger.warning(f"Could not detect layer path, using fallback: {fallback_path}")
        return fallback_path
    
    def _detect_embedding_path(self, model: Optional[nn.Module], config) -> str:
        """Detect path to embedding layer.""" 
        if model is None:
            return self._get_fallback_embedding_path(config)
        
        for pattern in self.EMBEDDING_PATH_PATTERNS:
            try:
                embeddings = self._navigate_to_attribute(model, pattern)
                if self._validate_embedding_layer(embeddings, config):
                    logger.debug(f"Found embeddings at: {pattern}")
                    return pattern
            except (AttributeError, ValueError):
                continue
        
        # Fallback
        fallback_path = self._get_fallback_embedding_path(config)
        logger.warning(f"Could not detect embedding path, using fallback: {fallback_path}")
        return fallback_path
    
    def _detect_lm_head_path(self, model: Optional[nn.Module], config) -> Optional[str]:
        """Detect path to language modeling head."""
        if model is None:
            return None
        
        for pattern in self.LM_HEAD_PATH_PATTERNS:
            try:
                lm_head = self._navigate_to_attribute(model, pattern)
                if lm_head is not None:
                    logger.debug(f"Found LM head at: {pattern}")
                    return pattern
            except (AttributeError, ValueError):
                continue
        
        logger.debug("No LM head found")
        return None
    
    def _navigate_to_attribute(self, obj: nn.Module, path: str) -> Any:
        """Navigate to nested attribute using dot notation."""
        attrs = path.split('.')
        current = obj
        
        for attr in attrs:
            if not hasattr(current, attr):
                raise AttributeError(f"No attribute '{attr}' in path '{path}'")
            current = getattr(current, attr)
        
        return current
    
    def _validate_transformer_layers(self, layers, config) -> bool:
        """Validate that found object is actually transformer layers."""
        try:
            # Check if it's iterable and has correct length
            if not hasattr(layers, '__len__') or not hasattr(layers, '__iter__'):
                return False
            
            expected_layers = getattr(config, 'num_hidden_layers', getattr(config, 'num_layers', 12))
            if len(layers) != expected_layers:
                return False
            
            # Check if first layer looks like a transformer block
            if len(layers) > 0:
                first_layer = layers[0]
                # Look for attention-related components
                layer_attrs = [name for name, _ in first_layer.named_modules()]
                has_attention = any('attention' in attr.lower() or 'attn' in attr.lower() 
                                 for attr in layer_attrs)
                if not has_attention:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _validate_embedding_layer(self, embeddings, config) -> bool:
        """Validate that found object is an embedding layer."""
        try:
            # Check if it's an embedding or has embedding-like attributes
            if isinstance(embeddings, nn.Embedding):
                return True
            
            # Check for embedding-related attributes
            if hasattr(embeddings, 'word_embeddings') or hasattr(embeddings, 'token_embeddings'):
                return True
            
            # Check if it has weight tensor with vocab_size
            if hasattr(embeddings, 'weight'):
                vocab_size = getattr(config, 'vocab_size', None)
                if vocab_size and embeddings.weight.shape[0] == vocab_size:
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _introspect_layer_path(self, model: nn.Module, config) -> Optional[str]:
        """Try to find transformer layers through introspection."""
        expected_layers = getattr(config, 'num_hidden_layers', getattr(config, 'num_layers', 12))
        
        # Search through all named modules
        for name, module in model.named_modules():
            if hasattr(module, '__len__') and len(module) == expected_layers:
                # Check if it contains transformer-like layers
                if self._validate_transformer_layers(module, config):
                    return name
        
        return None
    
    def _get_fallback_layer_path(self, architecture_type: str) -> str:
        """Get fallback layer path based on architecture type."""
        fallback_mapping = {
            'gpt2': 'transformer.h',
            'gpt_neo': 'transformer.h',
            'gptj': 'transformer.h',
            'llama': 'model.layers',
            'mistral': 'model.layers',
            'falcon': 'transformer.h',
            'mpt': 'transformer.blocks',
            'opt': 'model.decoder.layers',
            'bloom': 'transformer.h',
            'bert': 'encoder.layer',
            'roberta': 'encoder.layer',
            'distilbert': 'transformer.layer',
            'electra': 'encoder.layer',
            'deberta': 'encoder.layer',
            'albert': 'encoder.layer',
            't5': 'encoder.block',
            'bart': 'model.encoder.layers',
            'vit': 'encoder.layer',
            'clip': 'encoder.layers',
        }
        
        return fallback_mapping.get(architecture_type, 'transformer.h')
    
    def _get_fallback_embedding_path(self, config) -> str:
        """Get fallback embedding path based on config."""
        architecture_type = self._detect_architecture_type(config)
        
        fallback_mapping = {
            'bert': 'embeddings',
            'roberta': 'embeddings', 
            'distilbert': 'embeddings',
            'gpt2': 'transformer.wte',
            'llama': 'model.embed_tokens',
            'mistral': 'model.embed_tokens',
            't5': 'shared',
            'vit': 'embeddings.patch_embeddings',
        }
        
        return fallback_mapping.get(architecture_type, 'embeddings')
    
    def _supports_causal_lm(self, config, architecture_family: str) -> bool:
        """Check if model supports causal language modeling."""
        return architecture_family in ["DECODER_ONLY", "ENCODER_DECODER"]
    
    def _supports_masked_lm(self, config, architecture_family: str) -> bool:
        """Check if model supports masked language modeling."""
        return architecture_family in ["ENCODER_ONLY", "ENCODER_DECODER"]
    
    def _requires_remote_code(self, model_name: str, config) -> bool:
        """Check if model requires remote code execution."""
        try:
            # Try to load without remote code
            AutoConfig.from_pretrained(model_name, trust_remote_code=False)
            AutoModel.from_pretrained(
                model_name, 
                trust_remote_code=False,
                torch_dtype=torch.float32,
                device_map="cpu"
            )
            return False
        except Exception:
            return True
    
    def get_layers(self, model: nn.Module, arch_info: ArchitectureInfo) -> nn.ModuleList:
        """Get transformer layers from model using detected path."""
        try:
            layers = self._navigate_to_attribute(model, arch_info.layer_path)
            if not self._validate_transformer_layers(layers, None):
                raise ValueError(f"Invalid layers at path: {arch_info.layer_path}")
            return layers
        except Exception as e:
            raise ValueError(f"Could not access layers at {arch_info.layer_path}: {e}")
    
    def get_embeddings(self, model: nn.Module, arch_info: ArchitectureInfo) -> nn.Module:
        """Get embedding layer from model using detected path."""
        try:
            embeddings = self._navigate_to_attribute(model, arch_info.embedding_path)
            return embeddings
        except Exception as e:
            raise ValueError(f"Could not access embeddings at {arch_info.embedding_path}: {e}")
    
    def get_lm_head(self, model: nn.Module, arch_info: ArchitectureInfo) -> Optional[nn.Module]:
        """Get LM head from model using detected path."""
        if not arch_info.lm_head_path:
            return None
        
        try:
            lm_head = self._navigate_to_attribute(model, arch_info.lm_head_path)
            return lm_head
        except Exception as e:
            logger.warning(f"Could not access LM head at {arch_info.lm_head_path}: {e}")
            return None
    
    def clear_cache(self):
        """Clear architecture detection cache."""
        self.architecture_cache.clear()
        logger.info("Cleared architecture detection cache")


# Global instance for easy access
universal_handler = UniversalArchitectureHandler()