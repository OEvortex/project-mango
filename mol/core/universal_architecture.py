"""
Universal Architecture Handler for MoL System

Uses the same method as mergekit and vLLM: relies completely on transformers'
native architecture detection via config.model_type and MODEL_MAPPING registry.
No hardcoded architecture mappings - trusts transformers completely.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
from transformers import (
    AutoModel, AutoConfig, AutoTokenizer,
    AutoModelForCausalLM, AutoModelForMaskedLM,
    MODEL_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING, MODEL_FOR_MASKED_LM_MAPPING
)
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
    
    Uses the same method as mergekit and vLLM: relies completely on 
    transformers' native architecture detection and MODEL_MAPPING registry.
    
    Key features:
    - No hardcoded architecture mappings
    - Uses config.model_type as primary identifier (like mergekit/vLLM)
    - Leverages transformers' MODEL_MAPPING for capabilities detection
    - Supports any model that transformers supports
    - Secure by default (trust_remote_code=False)
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
        
        Uses the same approach as mergekit and vLLM: trusts transformers
        completely for architecture detection via MODEL_MAPPING registry.
        
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
        Detect architecture using mergekit/vLLM method.
        
        Relies completely on transformers' native detection via:
        1. config.model_type (primary, same as mergekit/vLLM)
        2. config.architectures (fallback)
        3. MODEL_MAPPING registry for capabilities
        
        Args:
            model_name: HuggingFace model identifier
            
        Returns:
            ArchitectureInfo object with detected information
        """
        # Check cache first
        if model_name in self.architecture_cache:
            return self.architecture_cache[model_name]
        
        logger.info(f"🔍 Detecting architecture for {model_name} (mergekit/vLLM method)")
        
        try:
            # Load model config (transformers' standard approach)
            config = AutoConfig.from_pretrained(
                model_name, 
                trust_remote_code=self.trust_remote_code
            )
            
            # Detect using transformers' exact method
            architecture_type = self._detect_architecture_type(config)
            architecture_family = self._classify_architecture_family(config, architecture_type)
            
            # Load model for component path detection
            model = self._load_model_for_introspection(model_name, config)
            
            # Detect component paths using transformers conventions
            layer_path = self._detect_layer_path(model, config)
            embedding_path = self._detect_embedding_path(model, config)
            lm_head_path = self._detect_lm_head_path(model, config)
            
            # Extract configuration details using transformers' standard attributes
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
        """Detect architecture type using transformers' exact method (same as mergekit/vLLM)."""
        # Primary method: Use model_type (what transformers uses internally)
        if hasattr(config, 'model_type') and config.model_type:
            logger.debug(f"Detected architecture via model_type: {config.model_type}")
            return config.model_type.lower()
        
        # Fallback: Use architectures list (mergekit/vLLM fallback method)
        if hasattr(config, 'architectures') and config.architectures:
            arch_name = config.architectures[0]
            # Extract model_type from architecture name (transformers convention)
            clean_name = re.sub(r'(For\w+|Model)$', '', arch_name, flags=re.IGNORECASE)
            logger.debug(f"Detected architecture via architectures: {clean_name}")
            return clean_name.lower()
        
        # This should rarely happen with proper transformers models
        config_class = config.__class__.__name__
        if config_class.endswith('Config'):
            base_name = config_class[:-6]
            logger.warning(f"Fallback architecture detection from config class: {base_name}")
            return base_name.lower()
        
        raise ValueError(f"Could not detect architecture type from config: {config_class}")
    
    def _classify_architecture_family(self, config, architecture_type: str) -> str:
        """Classify architecture using transformers MODEL_MAPPING (mergekit/vLLM method)."""
        try:
            # Use transformers' native MODEL_MAPPING to determine capabilities
            # This is the exact method used by mergekit and vLLM
            config_class = config.__class__
            
            # Check if this config is in transformers' registries
            supports_causal = config_class in MODEL_FOR_CAUSAL_LM_MAPPING
            supports_masked = config_class in MODEL_FOR_MASKED_LM_MAPPING
            supports_base = config_class in MODEL_MAPPING
            
            logger.debug(f"Architecture {architecture_type}: causal={supports_causal}, masked={supports_masked}, base={supports_base}")
            
            # Determine family based on transformers' own knowledge
            if supports_causal and not supports_masked:
                return "DECODER_ONLY"
            elif supports_masked and not supports_causal:
                return "ENCODER_ONLY"
            elif supports_causal and supports_masked:
                return "ENCODER_DECODER"
            elif supports_base:
                # Special architectures (vision, multimodal, etc.)
                arch_lower = architecture_type.lower()
                if any(pattern in arch_lower for pattern in ['vit', 'deit', 'swin', 'beit', 'convnext']):
                    return "VISION"
                elif any(pattern in arch_lower for pattern in ['clip', 'flava', 'layoutlm', 'blip']):
                    return "MULTIMODAL"
                else:
                    return "OTHER"
            else:
                # Not in any transformers mapping - might be custom model
                logger.warning(f"Architecture {architecture_type} not found in transformers MODEL_MAPPING")
                return "DECODER_ONLY"  # Safe default
            
        except Exception as e:
            logger.warning(f"Error using MODEL_MAPPING for {architecture_type}: {e}")
            return "DECODER_ONLY"
    
    def _test_model_support(self, model_class, config) -> bool:
        """Test model support using transformers' MODEL_MAPPING (mergekit/vLLM approach)."""
        try:
            # Use transformers' own mapping registries
            config_class = config.__class__
            
            if model_class == AutoModelForCausalLM:
                return config_class in MODEL_FOR_CAUSAL_LM_MAPPING
            elif model_class == AutoModelForMaskedLM:
                return config_class in MODEL_FOR_MASKED_LM_MAPPING
            elif model_class == AutoModel:
                return config_class in MODEL_MAPPING
            
            # For other model classes, try direct mapping check
            if hasattr(model_class, '_model_mapping'):
                return config_class in model_class._model_mapping
            
            return False
            
        except Exception as e:
            logger.debug(f"Error checking model support: {e}")
            return False
    
    def _load_model_for_introspection(self, model_name: str, config) -> nn.Module:
        """Load model for architecture introspection with better error handling."""
        # Try different model loading strategies
        load_strategies = [
            # Try AutoModel first (most general)
            lambda: AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=self.trust_remote_code
            ),
            # Try AutoModelForCausalLM for decoder-only models
            lambda: AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=self.trust_remote_code
            ),
            # Try config-only approach for problematic models
            lambda: self._create_model_from_config(model_name, config)
        ]
        
        for i, strategy in enumerate(load_strategies):
            try:
                logger.debug(f"Trying model loading strategy {i+1}")
                model = strategy()
                if model is not None:
                    logger.debug(f"Successfully loaded model with strategy {i+1}")
                    return model
            except Exception as e:
                logger.debug(f"Model loading strategy {i+1} failed: {e}")
                continue
        
        logger.warning(f"All model loading strategies failed for {model_name}")
        return None
    
    def _create_model_from_config(self, model_name: str, config) -> Optional[nn.Module]:
        """Create a minimal model instance from config for introspection."""
        try:
            # Import the model class based on config
            from transformers import MODEL_MAPPING
            if config.__class__ in MODEL_MAPPING:
                model_class = MODEL_MAPPING[config.__class__]
                # Create model with minimal resources
                model = model_class(config)
                return model
            return None
        except Exception as e:
            logger.debug(f"Could not create model from config: {e}")
            return None
    
    def _detect_layer_path(self, model: Optional[nn.Module], config) -> str:
        """Detect path to transformer layers using pure introspection (vLLM/mergekit approach)."""
        if model is None:
            raise ValueError("Cannot detect architecture without loading model. Model loading failed.")
        
        # Use pure introspection to find transformer layers
        layer_path = self._introspect_layer_path(model, config)
        if layer_path:
            logger.debug(f"Found transformer layers at: {layer_path}")
            return layer_path
        
        # If introspection fails completely, this is an error
        raise ValueError(
            f"Could not detect transformer layers in model {model.__class__.__name__}. "
            f"Model may not be supported or may require trust_remote_code=True."
        )
    

    
    def _detect_embedding_path(self, model: Optional[nn.Module], config) -> str:
        """Detect path to embedding layer using pure introspection."""
        if model is None:
            raise ValueError("Cannot detect embeddings without loading model. Model loading failed.")
        
        # First, try to find embeddings through introspection
        embedding_path = self._introspect_embedding_path(model, config)
        if embedding_path:
            logger.debug(f"Found embeddings at: {embedding_path}")
            return embedding_path
        
        # If pure introspection fails, this is an error
        raise ValueError(
            f"Could not detect embeddings in model {model.__class__.__name__}. "
            f"Model may not be supported or may require trust_remote_code=True."
        )
    
    def _introspect_embedding_path(self, model: nn.Module, config) -> Optional[str]:
        """Find embeddings through comprehensive model introspection."""
        vocab_size = getattr(config, 'vocab_size', None)
        hidden_size = getattr(config, 'hidden_size', getattr(config, 'd_model', None))
        
        logger.debug(f"Looking for embeddings with vocab_size={vocab_size}, hidden_size={hidden_size}")
        
        candidates = []
        
        # Search through all named modules for embedding-like layers
        for name, module in model.named_modules():
            # Check for explicit nn.Embedding layers
            if isinstance(module, nn.Embedding) and vocab_size:
                if module.num_embeddings == vocab_size:
                    candidates.append((name, 'embedding', len(name.split('.'))))
                    logger.debug(f"Found nn.Embedding: {name} ({module.num_embeddings} tokens)")
            
            # Check for linear layers that could be embeddings (embedding tables)
            elif isinstance(module, nn.Linear) and vocab_size and hidden_size:
                if (module.in_features == vocab_size and module.out_features == hidden_size) or \
                   (module.out_features == vocab_size and module.in_features == hidden_size):
                    candidates.append((name, 'linear', len(name.split('.'))))
                    logger.debug(f"Found Linear embedding-like layer: {name} ({module.in_features}->{module.out_features})")
            
            # Check for modules with embedding-like weight tensors
            elif hasattr(module, 'weight') and vocab_size:
                try:
                    weight = module.weight
                    if hasattr(weight, 'shape') and len(weight.shape) == 2:
                        if weight.shape[0] == vocab_size or (hidden_size and weight.shape == (vocab_size, hidden_size)):
                            candidates.append((name, 'weight', len(name.split('.'))))
                            logger.debug(f"Found weight-based embedding: {name} {weight.shape}")
                except Exception:
                    continue
        
        if candidates:
            # Prefer actual nn.Embedding layers, then by path depth (shorter = closer to root)
            candidates.sort(key=lambda x: (x[1] != 'embedding', x[2], x[0]))
            best_candidate = candidates[0]
            logger.debug(f"Selected best embedding candidate: {best_candidate[0]}")
            return best_candidate[0]
        
        # Fallback: Look for common embedding patterns
        embedding_patterns = ['embed', 'embedding', 'word_embed', 'token_embed', 'wte', 'shared']
        
        for name, module in model.named_modules():
            name_lower = name.lower()
            for pattern in embedding_patterns:
                if pattern in name_lower and hasattr(module, 'weight'):
                    try:
                        weight = module.weight
                        if hasattr(weight, 'shape') and len(weight.shape) == 2:
                            if vocab_size and weight.shape[0] == vocab_size:
                                logger.debug(f"Found embedding by pattern: {name}")
                                return name
                    except Exception:
                        continue
        
        logger.warning("Could not find embeddings through introspection")
        return None
    

    
    def _detect_lm_head_path(self, model: Optional[nn.Module], config) -> Optional[str]:
        """Detect path to language modeling head using introspection."""
        if model is None:
            return None
        
        # First try introspection to find LM head
        lm_head_path = self._introspect_lm_head_path(model, config)
        if lm_head_path:
            logger.debug(f"Found LM head at: {lm_head_path}")
            return lm_head_path
        
        # Try common patterns as fallback
        for pattern in self.LM_HEAD_PATH_PATTERNS:
            try:
                lm_head = self._navigate_to_attribute(model, pattern)
                if lm_head is not None:
                    logger.debug(f"Found LM head at pattern: {pattern}")
                    return pattern
            except (AttributeError, ValueError):
                continue
        
        logger.debug("No LM head found")
        return None
    
    def _introspect_lm_head_path(self, model: nn.Module, config) -> Optional[str]:
        """Find LM head through model introspection."""
        vocab_size = getattr(config, 'vocab_size', None)
        hidden_size = getattr(config, 'hidden_size', getattr(config, 'd_model', None))
        
        # Look for linear layers that map from hidden_size to vocab_size
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and vocab_size and hidden_size:
                if (module.in_features == hidden_size and 
                    module.out_features == vocab_size):
                    return name
            
            # Also check for modules that might contain such layers
            if hasattr(module, 'weight') and vocab_size and hidden_size:
                try:
                    weight = module.weight
                    if hasattr(weight, 'shape') and len(weight.shape) == 2:
                        if (weight.shape[0] == vocab_size and 
                            weight.shape[1] == hidden_size):
                            return name
                except Exception:
                    continue
        
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
            
            if config is not None:
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
    
    def _lenient_validate_transformer_layers(self, layers, arch_info: ArchitectureInfo) -> bool:
        """More lenient validation that focuses on structure rather than exact count."""
        try:
            # Check basic properties
            if not hasattr(layers, '__len__') or not hasattr(layers, '__iter__'):
                return False
            
            if len(layers) == 0:
                return False
            
            # Check if we have a reasonable number of layers (not exact match required)
            layer_count = len(layers)
            expected_count = arch_info.num_layers
            
            # Allow some flexibility in layer count (within 50% of expected)
            if expected_count > 0:
                ratio = layer_count / expected_count
                if ratio < 0.5 or ratio > 2.0:
                    logger.debug(f"Layer count mismatch: found {layer_count}, expected ~{expected_count}")
                    return False
            
            # Check if it contains transformer-like modules
            return self._contains_attention_modules(layers)
            
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
        """Find transformer layers through comprehensive model introspection."""
        expected_layers = getattr(config, 'num_hidden_layers', getattr(config, 'num_layers', None))
        
        logger.debug(f"Looking for {expected_layers} transformer layers in model structure")
        
        # First pass: Look for ModuleList/Sequential with expected length
        candidates = []
        
        for name, module in model.named_modules():
            # Skip very short names (usually not layer containers)
            if len(name.split('.')) < 2:
                continue
                
            # Check if this could be a layer container
            if hasattr(module, '__len__'):
                try:
                    module_length = len(module)
                    if expected_layers and module_length == expected_layers:
                        # Validate that it contains transformer-like layers
                        if self._validate_transformer_layers(module, config):
                            candidates.append((name, module, module_length))
                            logger.debug(f"Found candidate layer container: {name} (length: {module_length})")
                except (TypeError, AttributeError):
                    continue
        
        # If we found candidates, pick the best one
        if candidates:
            # Prefer shorter paths (closer to root)
            candidates.sort(key=lambda x: (len(x[0].split('.')), x[0]))
            best_candidate = candidates[0]
            logger.debug(f"Selected best candidate: {best_candidate[0]}")
            return best_candidate[0]
        
        # Second pass: Look for any container with transformer-like modules
        logger.debug("No exact length match found, searching for transformer-like containers")
        
        for name, module in model.named_modules():
            if hasattr(module, '__len__') and len(module) > 0:
                try:
                    # Check if it contains attention-related modules
                    if self._contains_attention_modules(module):
                        logger.debug(f"Found attention-containing container: {name} (length: {len(module)})")
                        return name
                except (TypeError, AttributeError):
                    continue
        
        logger.warning("Could not find transformer layers through introspection")
        return None
    
    def _contains_attention_modules(self, container) -> bool:
        """Check if a container has attention-related modules."""
        try:
            if hasattr(container, '__iter__'):
                # Check first few items for attention patterns
                for i, item in enumerate(container):
                    if i >= 3:  # Only check first few
                        break
                    if self._has_attention_components(item):
                        return True
            return False
        except Exception:
            return False
    
    def _has_attention_components(self, module) -> bool:
        """Check if a module has attention-related components."""
        try:
            # Get all sub-module names
            submodule_names = [name.lower() for name, _ in module.named_modules()]
            
            # Look for attention-related terms
            attention_terms = ['attention', 'attn', 'self_attn', 'multi_head', 'mha']
            
            for term in attention_terms:
                if any(term in name for name in submodule_names):
                    return True
            
            # Also check for common transformer patterns
            transformer_terms = ['query', 'key', 'value', 'q_proj', 'k_proj', 'v_proj']
            attention_count = sum(1 for term in transformer_terms 
                                for name in submodule_names if term in name)
            
            return attention_count >= 2  # At least 2 attention-related components
            
        except Exception:
            return False
    
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
        """Check causal LM support using transformers MODEL_MAPPING (mergekit/vLLM method)."""
        try:
            # Direct check against transformers' registry
            return config.__class__ in MODEL_FOR_CAUSAL_LM_MAPPING
        except Exception:
            # Minimal fallback
            return architecture_family in ["DECODER_ONLY", "ENCODER_DECODER"]
    
    def _supports_masked_lm(self, config, architecture_family: str) -> bool:
        """Check masked LM support using transformers MODEL_MAPPING (mergekit/vLLM method)."""
        try:
            # Direct check against transformers' registry  
            return config.__class__ in MODEL_FOR_MASKED_LM_MAPPING
        except Exception:
            # Minimal fallback
            return architecture_family in ["ENCODER_ONLY", "ENCODER_DECODER"]
    
    def _requires_remote_code(self, model_name: str, config) -> bool:
        """Check if model requires remote code execution."""
        try:
            # Try to load without remote code
            from transformers import AutoConfig, AutoModel
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
            
            # Create a minimal config object for validation
            class MinimalConfig:
                def __init__(self, num_layers):
                    self.num_hidden_layers = num_layers
                    self.num_layers = num_layers
            
            minimal_config = MinimalConfig(arch_info.num_layers)
            
            if not self._validate_transformer_layers(layers, minimal_config):
                # If strict validation fails, do a more lenient check
                if not self._lenient_validate_transformer_layers(layers, arch_info):
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


# Global instance using mergekit/vLLM approach
universal_handler = UniversalArchitectureHandler()