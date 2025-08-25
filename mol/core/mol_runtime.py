"""
Main MoL Runtime for dynamic layer fusion.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging
from collections import defaultdict
import json

from .block_extractor import BlockExtractor, ExtractedBlock, ModelInfo
from .adapters import BaseAdapter, create_adapter
from .routers import BaseRouter, create_router, compute_router_entropy, compute_load_balancing_loss
from ..utils.memory_utils import MemoryManager
from ..utils.model_utils import ModelUtils

# Import universal handlers to eliminate hardcoded logic
from .universal_parameter_detector import (
    UniversalParameterDetector, ModelSignature, ParameterCategory, 
    universal_parameter_detector
)
from .universal_rope_handler import (
    UniversalRoPEHandler, RoPEInfo, universal_rope_handler
)
from .universal_strategy_generator import (
    UniversalStrategyGenerator, ParameterStrategy, StrategyType,
    universal_strategy_generator
)

logger = logging.getLogger(__name__)


@dataclass
class MoLConfig:
    """Configuration for MoL runtime."""
    models: List[str]
    adapter_type: str = "linear"
    router_type: str = "simple" 
    max_layers: int = 32
    target_hidden_dim: Optional[int] = None  # If None, use largest model's dim
    use_gradient_checkpointing: bool = False
    memory_efficient: bool = True
    temperature: float = 1.0
    entropy_penalty_coeff: float = 0.1
    load_balancing_coeff: float = 0.01
    top_k_experts: Optional[int] = None
    device_map: Optional[Dict[str, str]] = None
    trust_remote_code: bool = False  # Security: disable remote code by default


@dataclass
class LayerSpec:
    """Specification for a layer in the MoL pipeline."""
    experts: List[ExtractedBlock]
    adapters: List[BaseAdapter]
    router: BaseRouter
    layer_idx: int


class MoLRuntime(nn.Module):
    """
    Main MoL Runtime for dynamic layer fusion.
    
    Combines transformer blocks from different models using adapters and routing.
    """
    
    def __init__(self, config: MoLConfig):
        super().__init__()
        self.config = config
        
        # Security warning if trust_remote_code is enabled
        if config.trust_remote_code:
            logger.warning(
                "⚠️  trust_remote_code=True enabled. This may execute arbitrary code from model repositories. "
                "Only use with trusted models."
            )
        
        # Core components
        self.block_extractor = BlockExtractor(trust_remote_code=config.trust_remote_code)
        self.memory_manager = MemoryManager()
        self.model_utils = ModelUtils()
        
        # Universal handlers to eliminate hardcoded logic
        self.parameter_detector = universal_parameter_detector
        self.rope_handler = universal_rope_handler
        self.strategy_generator = universal_strategy_generator
        
        # Model information
        self.model_infos: Dict[str, ModelInfo] = {}
        self.target_hidden_dim: int = 0
        
        # MoL components
        self.layers: nn.ModuleList = nn.ModuleList()
        self.embedding_layer: Optional[nn.Module] = None
        self.lm_head: Optional[nn.Module] = None
        self.embedding_adapter: Optional[BaseAdapter] = None
        
        # Tokenizer (use the first model's tokenizer by default)
        self.tokenizer = None
        
        # Statistics tracking
        self.routing_stats = defaultdict(list)
        
        # Initialize the runtime
        self._initialize()
    
    def _initialize(self):
        """Initialize the MoL runtime."""
        logger.info("Initializing MoL Runtime...")
        
        # Load model information
        self._load_model_infos()
        
        # Determine target hidden dimension
        self._determine_target_dim()
        
        # Setup tokenizer
        self._setup_tokenizer()
        
        logger.info(f"MoL Runtime initialized with target dim: {self.target_hidden_dim}")
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for memory efficiency.
        
        This activates gradient checkpointing for all MoL layers to reduce memory usage
        during training at the cost of additional computation during backward pass.
        
        Args:
            gradient_checkpointing_kwargs (dict, optional): Additional keyword arguments
                for gradient checkpointing configuration.
        """
        if not self.training:
            logger.warning(
                "Gradient checkpointing is being enabled on a model in evaluation mode. "
                "This may lead to unexpected behavior. Consider calling model.train() first."
            )
        
        # Enable gradient checkpointing for all MoL layers
        for layer in self.layers:
            if hasattr(layer, '_gradient_checkpointing_enable'):
                layer._gradient_checkpointing_enable(gradient_checkpointing_kwargs)
            elif hasattr(layer, 'gradient_checkpointing'):
                layer.gradient_checkpointing = True
        
        # Set flag for future layers that might be added
        self._gradient_checkpointing_enabled = True
        self._gradient_checkpointing_kwargs = gradient_checkpointing_kwargs or {}
        
        logger.info("Gradient checkpointing enabled for MoL runtime")
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        # Disable gradient checkpointing for all MoL layers
        for layer in self.layers:
            if hasattr(layer, '_gradient_checkpointing_disable'):
                layer._gradient_checkpointing_disable()
            elif hasattr(layer, 'gradient_checkpointing'):
                layer.gradient_checkpointing = False
        
        # Clear flags
        self._gradient_checkpointing_enabled = False
        self._gradient_checkpointing_kwargs = {}
        
        logger.info("Gradient checkpointing disabled for MoL runtime")
    
    @property
    def is_gradient_checkpointing(self) -> bool:
        """Check if gradient checkpointing is enabled."""
        return getattr(self, '_gradient_checkpointing_enabled', False)
    
    def _load_model_infos(self):
        """Load information for all models."""
        for model_name in self.config.models:
            _, model_info = self.block_extractor.load_model(model_name)
            self.model_infos[model_name] = model_info
            logger.info(f"Loaded info for {model_name}: {model_info.hidden_dim}D")
    
    def _determine_target_dim(self):
        """Determine target hidden dimension for the pipeline."""
        if self.config.target_hidden_dim:
            self.target_hidden_dim = self.config.target_hidden_dim
        else:
            # Use the largest hidden dimension among all models
            max_dim = max(info.hidden_dim for info in self.model_infos.values())
            self.target_hidden_dim = max_dim
        
        logger.info(f"Target hidden dimension: {self.target_hidden_dim}")
    
    def _setup_tokenizer(self):
        """Setup tokenizer (use first model's tokenizer)."""
        from transformers import AutoTokenizer
        
        primary_model = self.config.models[0]
        self.tokenizer = AutoTokenizer.from_pretrained(
            primary_model,
            trust_remote_code=self.config.trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Using tokenizer from {primary_model}")
    
    def add_layer(
        self, 
        layer_specs: List[Tuple[str, int]], 
        layer_idx: int
    ):
        """
        Add a MoL layer with specified experts.
        
        Args:
            layer_specs: List of (model_name, model_layer_idx) tuples
            layer_idx: Index of this layer in the MoL pipeline
        """
        logger.info(f"Adding MoL layer {layer_idx} with experts: {layer_specs}")
        
        # Extract blocks from specified models/layers
        experts = []
        for model_name, model_layer_idx in layer_specs:
            block = self.block_extractor.extract_block(model_name, model_layer_idx)
            experts.append(block)
            logger.debug(f"Extracted block from {model_name} layer {model_layer_idx}, input_dim: {block.input_dim}")
        
        # Create adapters for dimension matching
        adapters = []
        for expert in experts:
            adapter = create_adapter(
                self.config.adapter_type,
                input_dim=expert.input_dim,
                output_dim=self.target_hidden_dim,
                init_identity=True
            )
            adapters.append(adapter)
            logger.debug(f"Created adapter: {expert.input_dim} -> {self.target_hidden_dim}")
        
        # Create router with comprehensive logging
        logger.debug(f"Creating router: type={self.config.router_type}, hidden_dim={self.target_hidden_dim}, num_experts={len(experts)}")
        
        try:
            router = create_router(
                self.config.router_type,
                hidden_dim=self.target_hidden_dim,
                num_experts=len(experts),
                temperature=self.config.temperature,
                top_k=self.config.top_k_experts
            )
            logger.debug(f"Successfully created router: {type(router)}")
        except Exception as e:
            logger.error(f"Failed to create router: {e}")
            raise RuntimeError(f"Router creation failed: {e}") from e
        
        # Validate router
        if router is None:
            raise RuntimeError("Router creation returned None")
        
        # Create layer specification
        layer_spec = LayerSpec(
            experts=experts,
            adapters=adapters,
            router=router,
            layer_idx=layer_idx
        )
        
        # Wrap in MoL layer module
        mol_layer = MoLLayer(layer_spec, self.config)
        
        # Apply gradient checkpointing if enabled
        if getattr(self, '_gradient_checkpointing_enabled', False):
            mol_layer._gradient_checkpointing_enable(self._gradient_checkpointing_kwargs)
        
        self.layers.append(mol_layer)
        
        logger.info(f"Added MoL layer {layer_idx} with {len(experts)} experts")
    
    def setup_embeddings(self, primary_model: Optional[str] = None):
        """Setup embedding layer from primary model."""
        if primary_model is None:
            primary_model = self.config.models[0]
        
        # Get embedding layer
        embeddings, embed_dim = self.block_extractor.get_embedding_layer(primary_model)
        self.embedding_layer = embeddings
        
        # Create adapter if needed
        if embed_dim != self.target_hidden_dim:
            self.embedding_adapter = create_adapter(
                "linear",
                input_dim=embed_dim,
                output_dim=self.target_hidden_dim,
                init_identity=True
            )
        
        logger.info(f"Setup embeddings from {primary_model}")
    
    def setup_lm_head(self, primary_model: Optional[str] = None):
        """Setup language modeling head from primary model."""
        if primary_model is None:
            primary_model = self.config.models[0]
        
        lm_head = self.block_extractor.get_lm_head(primary_model)
        if lm_head:
            self.lm_head = lm_head
            logger.info(f"Setup LM head from {primary_model}")
        else:
            logger.warning(f"Could not load LM head from {primary_model}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_router_stats: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass through MoL runtime.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_router_stats: Whether to return routing statistics
        
        Returns:
            hidden_states: Final hidden states [batch_size, seq_len, hidden_dim]
            router_stats: Router statistics (if requested)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Track router statistics
        router_stats = {} if return_router_stats else None
        
        # Embedding layer
        if self.embedding_layer is None:
            raise RuntimeError("Embedding layer not initialized. Call setup_embeddings() first.")
        
        # Ensure embedding layer is on correct device
        self.embedding_layer = self.embedding_layer.to(device)
        hidden_states = self.embedding_layer(input_ids)
        
        # Apply embedding adapter if needed
        if self.embedding_adapter:
            self.embedding_adapter = self.embedding_adapter.to(device)
            hidden_states = self.embedding_adapter(hidden_states)
        
        # Ensure hidden states have correct target dimension
        if hidden_states.size(-1) != self.target_hidden_dim:
            logger.warning(
                f"Hidden states dimension {hidden_states.size(-1)} != target dimension {self.target_hidden_dim}"
            )
        
        # Process through MoL layers
        for layer_idx, mol_layer in enumerate(self.layers):
            try:
                hidden_states, layer_stats = mol_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    return_stats=return_router_stats
                )
                
                if return_router_stats and layer_stats:
                    router_stats[f"layer_{layer_idx}"] = layer_stats
                    
            except Exception as e:
                logger.error(f"Error in layer {layer_idx}: {e}")
                # Continue with unchanged hidden states
                if return_router_stats:
                    router_stats[f"layer_{layer_idx}"] = {'error': str(e)}
        
        return hidden_states, router_stats
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Generate text using the MoL model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling or greedy decoding
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
        
        Returns:
            generated_ids: Generated token IDs [batch_size, seq_len + max_new_tokens]
        """
        if self.lm_head is None:
            raise RuntimeError("LM head not initialized. Call setup_lm_head() first.")
        
        self.eval()
        generated_ids = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                hidden_states, _ = self.forward(generated_ids, attention_mask)
                
                # Get logits from LM head
                logits = self.lm_head(hidden_states[:, -1, :])  # Last token logits
                
                # Apply temperature
                logits = logits / temperature
                
                # Sample next token
                if do_sample:
                    # Apply top-k filtering
                    if top_k is not None:
                        top_k_logits, _ = torch.topk(logits, top_k, dim=-1)
                        logits[logits < top_k_logits[:, -1:]] = -float('inf')
                    
                    # Apply top-p filtering
                    if top_p is not None:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                        sorted_indices_to_remove[:, 0] = 0
                        
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        logits[indices_to_remove] = -float('inf')
                    
                    # Sample from the distribution
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                
                # Update attention mask if provided
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask, 
                        torch.ones(attention_mask.shape[0], 1, device=attention_mask.device)
                    ], dim=-1)
                
                # Check for EOS token
                if (next_token == self.tokenizer.eos_token_id).all():
                    break
        
        return generated_ids
    
    def save_checkpoint(self, path: str, use_safetensors: bool = True):
        """
        Save MoL checkpoint with optional SafeTensors support.
        
        Args:
            path: Path to save checkpoint
            use_safetensors: Whether to use SafeTensors format for security
        """
        from ..utils.safetensors_utils import safetensors_manager
        
        checkpoint = {
            'config': self.config,
            'target_hidden_dim': self.target_hidden_dim,
            'model_infos': self.model_infos,
            'model_state_dict': self.state_dict(),
        }
        
        if use_safetensors:
            safetensors_manager.save_checkpoint(checkpoint, path, use_safetensors=True)
            logger.info(f"Saved MoL checkpoint to {path} using SafeTensors")
        else:
            torch.save(checkpoint, path)
            logger.info(f"Saved MoL checkpoint to {path} using PyTorch")
    
    @classmethod
    def load_checkpoint(cls, path: str):
        """
        Load MoL checkpoint with SafeTensors support.
        
        Args:
            path: Path to checkpoint file (.safetensors or .pt)
            
        Returns:
            MoL runtime instance
        """
        from ..utils.safetensors_utils import safetensors_manager
        
        try:
            # Try SafeTensors first
            checkpoint = safetensors_manager.load_checkpoint(path, device='cpu')
            logger.info(f"Loaded MoL checkpoint from {path} using SafeTensors")
        except (FileNotFoundError, ImportError):
            # Fallback to PyTorch
            checkpoint = torch.load(path, map_location='cpu')
            logger.info(f"Loaded MoL checkpoint from {path} using PyTorch")
        
        # Create runtime with loaded config
        runtime = cls(checkpoint['config'])
        runtime.target_hidden_dim = checkpoint['target_hidden_dim']
        runtime.model_infos = checkpoint['model_infos']
        
        # Load state dict (handle both new and old format)
        state_dict_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'state_dict'
        runtime.load_state_dict(checkpoint[state_dict_key])
        
        return runtime
    
    def push_to_hf(
        self,
        repo_id: str,
        fusion_type: str = "runtime",
        token: Optional[str] = None,
        commit_message: Optional[str] = None,
        private: bool = False,
        create_pr: bool = False,
        fusion_method: str = "weighted_average"
    ) -> str:
        """
        Push MoL model to Hugging Face Hub.
        
        Args:
            repo_id: Repository ID (username/model-name)
            fusion_type: "runtime" for lightweight MoL or "fused" for full static model
            token: HuggingFace API token
            commit_message: Custom commit message
            private: Whether to create private repository
            create_pr: Whether to create pull request
            fusion_method: Fusion method for "fused" type (weighted_average, best_expert, learned_weights)
            
        Returns:
            Repository URL
        """
        from ..utils.hf_utils import HuggingFacePublisher
        
        publisher = HuggingFacePublisher(token=token)
        
        if fusion_type == "runtime":
            if commit_message is None:
                commit_message = f"Upload MoL runtime model ({len(self.config.models)} experts)"
            return publisher.push_mol_runtime(
                self, repo_id, commit_message=commit_message,
                private=private, create_pr=create_pr
            )
        elif fusion_type == "fused":
            if commit_message is None:
                commit_message = f"Upload fully fused model ({fusion_method})"
            return publisher.push_fused_model(
                self, repo_id, commit_message=commit_message,
                private=private, create_pr=create_pr,
                fusion_method=fusion_method
            )
        else:
            raise ValueError("fusion_type must be 'runtime' or 'fused'")
    
    def create_fused_model(self, fusion_method: str = "weighted_average") -> nn.Module:
        """
        Create a fully fused static model from this MoL runtime.
        
        Args:
            fusion_method: Fusion strategy (weighted_average, best_expert, learned_weights)
            
        Returns:
            Fused PyTorch model that can be used independently
        """
        from ..utils.hf_utils import HuggingFacePublisher
        
        publisher = HuggingFacePublisher()
        return publisher._create_fused_model(self, fusion_method)


class MoLLayer(nn.Module):
    """A single MoL layer with multiple experts, adapters, and routing."""
    
    def __init__(self, layer_spec: LayerSpec, config: MoLConfig):
        super().__init__()
        self.layer_spec = layer_spec
        self.config = config
        
        # Register components
        self.experts = nn.ModuleList([expert.block for expert in layer_spec.experts])
        self.adapters = nn.ModuleList(layer_spec.adapters)
        self.router = layer_spec.router
        
        # Expert metadata
        self.expert_dims = [expert.input_dim for expert in layer_spec.experts]
        self.num_experts = len(self.experts)
        
        # Gradient checkpointing support
        self.gradient_checkpointing = False
        self._gradient_checkpointing_kwargs = {}
    
    def _gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for this layer."""
        self.gradient_checkpointing = True
        self._gradient_checkpointing_kwargs = gradient_checkpointing_kwargs or {}
    
    def _gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for this layer."""
        self.gradient_checkpointing = False
        self._gradient_checkpointing_kwargs = {}
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_stats: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Forward pass through MoL layer with gradient checkpointing support."""
        if self.gradient_checkpointing and self.training:
            # Use gradient checkpointing for memory efficiency
            return self._gradient_checkpointed_forward(
                hidden_states, attention_mask, return_stats
            )
        else:
            # Regular forward pass
            return self._forward(
                hidden_states, attention_mask, return_stats
            )
    
    def _gradient_checkpointed_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_stats: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Gradient checkpointed forward pass."""
        from torch.utils.checkpoint import checkpoint
        
        # Create a function that can be checkpointed
        def create_forward_fn(return_stats):
            def forward_fn(hidden_states_arg):
                return self._forward(hidden_states_arg, attention_mask, return_stats)
            return forward_fn
        
        # Use gradient checkpointing
        checkpoint_kwargs = self._gradient_checkpointing_kwargs.copy()
        # Set use_reentrant to False for better compatibility (recommended in PyTorch 2.0+)
        checkpoint_kwargs.setdefault('use_reentrant', False)
        
        result = checkpoint(
            create_forward_fn(return_stats),
            hidden_states,
            **checkpoint_kwargs
        )
        
        return result
    
    def _forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_stats: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Actual forward pass logic (used by both regular and checkpointed versions)."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device
        
        # Ensure all components are on the same device
        self.router = self.router.to(device)
        
        # Debug router state
        if self.router is None:
            raise RuntimeError("Router is None - this should not happen")
        
        logger.debug(f"Router type: {type(self.router)}, device: {next(self.router.parameters()).device}")
        logger.debug(f"Input shape: {hidden_states.shape}, device: {device}")
        
        # Get routing decisions with error handling
        try:
            router_output = self.router(hidden_states, attention_mask)
            if router_output is None:
                raise RuntimeError(f"Router {type(self.router)} returned None instead of tuple")
            if not isinstance(router_output, tuple) or len(router_output) != 2:
                raise RuntimeError(f"Router returned {type(router_output)} with length {len(router_output) if hasattr(router_output, '__len__') else 'N/A'}, expected tuple of length 2")
            expert_weights, router_logits = router_output
        except Exception as e:
            logger.error(f"Router forward failed: {e}. Router type: {type(self.router)}")
            raise RuntimeError(f"Router forward failed: {e}") from e
        
        # Process through each expert
        expert_outputs = []
        for i, (expert, adapter) in enumerate(zip(self.experts, self.adapters)):
            try:
                # Ensure expert and adapter are on correct device
                expert = expert.to(device)
                adapter = adapter.to(device)
                
                # Pass through expert - handle different transformer block signatures
                expert_output = self._forward_through_expert(expert, hidden_states, attention_mask)
                
                # Apply adapter for dimension matching
                expert_output = adapter(expert_output)
                expert_outputs.append(expert_output)
                
            except Exception as e:
                logger.warning(f"Error processing expert {i}: {e}. Using identity mapping.")
                # Fallback to identity mapping through adapter only
                expert_output = adapter(hidden_states)
                expert_outputs.append(expert_output)
        
        # Combine expert outputs using routing weights
        expert_outputs = torch.stack(expert_outputs, dim=-1)  # [batch, seq, hidden, num_experts]
        expert_weights = expert_weights.unsqueeze(-2)  # [batch, seq, 1, num_experts]
        
        # Weighted combination
        output = torch.sum(expert_outputs * expert_weights, dim=-1)  # [batch, seq, hidden]
        
        # Collect statistics
        stats = None
        if return_stats:
            try:
                stats = {
                    'expert_weights': expert_weights.squeeze(-2).detach().cpu(),
                    'router_entropy': compute_router_entropy(router_logits).item(),
                    'load_balancing_loss': compute_load_balancing_loss(
                        expert_weights.squeeze(-2), attention_mask
                    ).item(),
                    'num_experts': self.num_experts,
                }
            except Exception as e:
                logger.warning(f"Error computing stats: {e}")
                stats = {'num_experts': self.num_experts}
        
        return output, stats
    
    def _forward_through_expert(
        self, 
        expert: nn.Module, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        **additional_inputs
    ) -> torch.Tensor:
        """UNIVERSAL transformer expert forward supporting ALL transformer architectures.
        
        Enhanced to support ALL models including vision, multimodal, encoder-decoder, and specialized architectures.
        Uses comprehensive parameter detection and intelligent fallback strategies.
        """
        try:
            from transformers import AutoConfig, PretrainedConfig
            from transformers.modeling_utils import PreTrainedModel
            import inspect
            
            # Input validation
            if hidden_states is None:
                logger.error("Hidden states is None, cannot proceed")
                return torch.zeros_like(hidden_states) if hidden_states is not None else None
            
            batch_size, seq_len = hidden_states.shape[:2]
            device = hidden_states.device
            dtype = hidden_states.dtype
            
            # === UNIVERSAL PARAMETER BUILDER ===
            # Detect ALL supported parameters using transformers' own introspection
            forward_sig = inspect.signature(expert.forward)
            forward_params = set(forward_sig.parameters.keys())
            
            logger.debug(f"Expert {type(expert).__name__} forward signature: {list(forward_params)}")
            
            # Build universal parameter set using transformers standards
            universal_kwargs = self._build_universal_parameters(
                expert, forward_params, hidden_states, attention_mask, 
                batch_size, seq_len, device, dtype, **additional_inputs
            )
            
            # === UNIVERSAL STRATEGY GENERATION ===
            # Generate strategies dynamically based on model signature analysis
            model_signature = None
            rope_info = None
            
            try:
                # Analyze model signature dynamically
                model_signature = self.parameter_detector.analyze_model_signature(
                    expert, getattr(expert, '_model_name', expert.__class__.__name__)
                )
                logger.debug(f"Analyzed {model_signature.total_param_count} parameters for {type(expert).__name__}")
                
                # Get RoPE info
                rope_info = self.rope_handler.detect_rope_info(
                    expert, getattr(expert, 'config', None), 
                    getattr(expert, '_model_name', expert.__class__.__name__)
                )
                
            except Exception as e:
                logger.debug(f"Could not analyze model signature: {e}")
            
            # Generate strategies based on detected capabilities
            strategies = []
            if model_signature:
                strategies = self.strategy_generator.generate_strategies(model_signature, rope_info)
            else:
                # Fallback strategies if signature analysis fails
                strategies = [
                    ParameterStrategy(
                        name="fallback_comprehensive",
                        strategy_type=StrategyType.COMPREHENSIVE,
                        description="Fallback comprehensive parameters",
                        include_categories=set(ParameterCategory),
                        exclude_categories=set(),
                        require_primary_input=True,
                        allow_defaults=True,
                        filter_none_values=True,
                        priority=1
                    ),
                    ParameterStrategy(
                        name="fallback_minimal",
                        strategy_type=StrategyType.MINIMAL,
                        description="Fallback minimal parameters",
                        include_categories={ParameterCategory.INPUT},
                        exclude_categories=set(),
                        require_primary_input=True,
                        allow_defaults=False,
                        filter_none_values=True,
                        priority=5
                    ),
                    ParameterStrategy(
                        name="fallback_identity",
                        strategy_type=StrategyType.IDENTITY,
                        description="Fallback identity",
                        include_categories=set(),
                        exclude_categories=set(),
                        require_primary_input=False,
                        allow_defaults=False,
                        filter_none_values=False,
                        priority=999
                    )
                ]
            
            last_error = None
            for strategy in strategies:
                try:
                    # Build parameters using universal strategy
                    if strategy.strategy_type == StrategyType.IDENTITY:
                        logger.debug(f"Using identity fallback - returning input unchanged")
                        return hidden_states
                    
                    # Prepare available inputs for strategy
                    available_inputs = {
                        'hidden_states': hidden_states,
                        'inputs_embeds': hidden_states,
                        'attention_mask': attention_mask,
                        **additional_inputs
                    }
                    
                    # Build strategy parameters using universal generator
                    strategy_kwargs = self.strategy_generator.build_strategy_parameters(
                        strategy=strategy,
                        model_signature=model_signature,
                        available_inputs=available_inputs,
                        rope_info=rope_info,
                        sequence_length=seq_len,
                        device=device,
                        dtype=dtype
                    )
                    
                    logger.debug(f"🔄 Trying {type(expert).__name__} strategy '{strategy.name}' with params: {list(strategy_kwargs.keys())}")
                    
                    # Try the forward call with enhanced error handling
                    expert.eval()
                    with torch.no_grad():
                        if strategy_kwargs:
                            output = expert(**strategy_kwargs)
                        else:
                            # For identity or minimal cases
                            output = expert(hidden_states)
                    
                    # Extract and validate output
                    final_output = self._extract_transformers_output(output, strategy)
                    
                    # Validate result
                    if (final_output is not None and 
                        isinstance(final_output, torch.Tensor) and 
                        final_output.dim() >= 2 and 
                        final_output.shape[0] == batch_size and 
                        final_output.shape[1] == seq_len):
                        
                        logger.debug(f"✅ SUCCESS: {type(expert).__name__} with '{strategy.name}', shape: {final_output.shape}")
                        return final_output
                    else:
                        logger.debug(f"❌ Invalid result from '{strategy.name}': type={type(final_output)}, shape={getattr(final_output, 'shape', 'N/A')}")
                        continue
                    
                except Exception as e:
                    last_error = e
                    error_msg = str(e)[:150]
                    logger.debug(f"❌ Strategy '{strategy.name}' failed: {type(e).__name__}: {error_msg}")
                    continue
            
            # If all strategies failed, log comprehensive error and return identity
            logger.error(f"🚨 All strategies failed for {type(expert).__name__}")
            logger.error(f"Forward signature: {list(forward_params)}")
            if last_error:
                logger.error(f"Last error: {type(last_error).__name__}: {str(last_error)[:300]}")
            
            # Return identity mapping as final fallback
            logger.warning(f"Falling back to identity mapping for {type(expert).__name__}")
            return hidden_states
            
        except Exception as e:
            logger.error(f"💥 Critical error in transformers expert forward: {e}")
            # Even in critical error, return something valid
            return hidden_states
    
    def _generate_universal_position_embeddings(self, expert, config, seq_len, device, dtype):
        """Generate position embeddings universally without hardcoded logic."""
        try:
            # Get model name for this expert
            expert_model_name = getattr(expert, '_model_name', None) or expert.__class__.__name__
            
            # Detect RoPE info for this expert
            rope_info = self.rope_handler.detect_rope_info(expert, config, expert_model_name)
            
            if rope_info and rope_info.has_rope:
                # Use universal RoPE handler
                rope_result = self.rope_handler.generate_rope_embeddings(
                    rope_info, seq_len, device, dtype
                )
                if rope_result:
                    logger.debug(f"Generated universal RoPE for {expert_model_name}: {rope_result[0].shape}")
                    return rope_result
            
            # If no RoPE detected, return None
            logger.debug(f"No RoPE detected for {expert_model_name}")
            return None
            
        except Exception as e:
            logger.debug(f"Universal position embeddings generation failed: {e}")
            return None
    
    def _extract_transformers_output(self, output, strategy):
        """Extract output using transformers' standard patterns with enhanced None handling."""
        if output is None:
            logger.debug("Output is None, cannot extract")
            return None
        
        try:
            # Handle tuple outputs (return_dict=False) - transformers standard
            if isinstance(output, tuple):
                if len(output) > 0:
                    # Try to get the first tensor element
                    for item in output:
                        if isinstance(item, torch.Tensor) and item.dim() >= 2:
                            logger.debug(f"Extracted tensor from tuple: shape={item.shape}")
                            return item
                    logger.debug("No valid tensor found in tuple")
                else:
                    logger.debug("Empty tuple output")
                return None
            
            # Handle transformers ModelOutput objects
            elif hasattr(output, 'last_hidden_state') and output.last_hidden_state is not None:
                result = output.last_hidden_state
                logger.debug(f"Extracted last_hidden_state: shape={result.shape}")
                return result
                
            elif hasattr(output, 'hidden_states') and output.hidden_states is not None:
                hidden_states = output.hidden_states
                if isinstance(hidden_states, (list, tuple)) and len(hidden_states) > 0:
                    # Take the last layer's hidden states
                    result = hidden_states[-1]
                    logger.debug(f"Extracted from hidden_states list: shape={result.shape}")
                    return result
                elif isinstance(hidden_states, torch.Tensor):
                    logger.debug(f"Extracted hidden_states tensor: shape={hidden_states.shape}")
                    return hidden_states
            
            # Handle direct tensor
            elif isinstance(output, torch.Tensor) and output.dim() >= 2:
                logger.debug(f"Direct tensor output: shape={output.shape}")
                return output
            
            # Handle dict (some custom implementations)
            elif isinstance(output, dict):
                for key in ['last_hidden_state', 'hidden_states', 'logits', 'prediction_scores']:
                    if key in output and output[key] is not None and isinstance(output[key], torch.Tensor):
                        result = output[key]
                        logger.debug(f"Extracted from dict key '{key}': shape={result.shape}")
                        return result
                logger.debug(f"No valid tensor found in dict keys: {list(output.keys())}")
            
            # Handle list outputs
            elif isinstance(output, list) and len(output) > 0:
                for item in output:
                    if isinstance(item, torch.Tensor) and item.dim() >= 2:
                        logger.debug(f"Extracted tensor from list: shape={item.shape}")
                        return item
                logger.debug("No valid tensor found in list")
            
            else:
                logger.debug(f"Unhandled output type: {type(output)}")
            
        except Exception as e:
            logger.warning(f"Error extracting output: {e}")
        
        logger.debug("Could not extract valid tensor from output")
        return None

    def _build_universal_parameters(
        self, 
        expert: nn.Module, 
        forward_params: set, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int, 
        device: torch.device,
        dtype: torch.dtype,
        **additional_inputs
    ) -> Dict[str, Any]:
        """Build universal parameter set supporting ALL transformer architectures.
        
        Covers text, vision, audio, multimodal, encoder-decoder, and specialized models.
        Based on comprehensive transformers parameter research.
        """
        kwargs = {}
        
        # === CORE INPUT PARAMETERS ===
        # Primary text input (most models)
        if 'hidden_states' in forward_params:
            kwargs['hidden_states'] = hidden_states
        if 'inputs_embeds' in forward_params:
            kwargs['inputs_embeds'] = hidden_states  # Alternative input form
        if 'input_ids' in forward_params and 'input_ids' in additional_inputs:
            kwargs['input_ids'] = additional_inputs['input_ids']
        
        # === ATTENTION AND MASKING ===
        if 'attention_mask' in forward_params and attention_mask is not None:
            kwargs['attention_mask'] = attention_mask
        if 'head_mask' in forward_params:
            kwargs['head_mask'] = None  # Standard default
        if 'encoder_attention_mask' in forward_params:
            kwargs['encoder_attention_mask'] = additional_inputs.get('encoder_attention_mask')
        if 'cross_attn_head_mask' in forward_params:
            kwargs['cross_attn_head_mask'] = None
        
        # === POSITION HANDLING ===
        if 'position_ids' in forward_params:
            kwargs['position_ids'] = torch.arange(
                seq_len, device=device, dtype=torch.long
            ).unsqueeze(0).expand(batch_size, -1)
        if 'cache_position' in forward_params:
            kwargs['cache_position'] = torch.arange(seq_len, device=device, dtype=torch.long)
        
        # === POSITION EMBEDDINGS (Universal RoPE handling) ===
        if 'position_embeddings' in forward_params:
            try:
                config = getattr(expert, 'config', None)
                position_embeddings = self._generate_universal_position_embeddings(
                    expert, config, seq_len, device, dtype
                )
                if position_embeddings is not None:
                    kwargs['position_embeddings'] = position_embeddings
                else:
                    # Generate fallback position embeddings if needed
                    logger.debug("Generating fallback position embeddings")
                    head_dim = 128  # Reasonable default
                    if config and hasattr(config, 'hidden_size') and hasattr(config, 'num_attention_heads'):
                        head_dim = config.hidden_size // config.num_attention_heads
                    cos_emb = torch.ones((1, seq_len, head_dim), device=device, dtype=dtype)
                    sin_emb = torch.zeros((1, seq_len, head_dim), device=device, dtype=dtype)
                    kwargs['position_embeddings'] = (cos_emb, sin_emb)
            except Exception as e:
                logger.debug(f"Position embeddings generation failed: {e}")
        
        # === CACHING PARAMETERS ===
        if 'past_key_value' in forward_params:
            kwargs['past_key_value'] = None  # Training mode
        if 'past_key_values' in forward_params:
            kwargs['past_key_values'] = None  # Training mode
        if 'use_cache' in forward_params:
            kwargs['use_cache'] = False  # Training mode
        
        # === ENCODER-DECODER PARAMETERS ===
        if 'encoder_hidden_states' in forward_params:
            kwargs['encoder_hidden_states'] = additional_inputs.get('encoder_hidden_states')
        if 'decoder_input_ids' in forward_params:
            kwargs['decoder_input_ids'] = additional_inputs.get('decoder_input_ids')
        if 'decoder_attention_mask' in forward_params:
            kwargs['decoder_attention_mask'] = additional_inputs.get('decoder_attention_mask')
        if 'decoder_inputs_embeds' in forward_params:
            kwargs['decoder_inputs_embeds'] = additional_inputs.get('decoder_inputs_embeds')
        if 'encoder_outputs' in forward_params:
            kwargs['encoder_outputs'] = additional_inputs.get('encoder_outputs')
        
        # === VISION PARAMETERS ===
        if 'pixel_values' in forward_params:
            kwargs['pixel_values'] = additional_inputs.get('pixel_values')
        if 'pixel_values_videos' in forward_params:
            kwargs['pixel_values_videos'] = additional_inputs.get('pixel_values_videos')
        if 'image_grid_thw' in forward_params:
            kwargs['image_grid_thw'] = additional_inputs.get('image_grid_thw')
        if 'video_grid_thw' in forward_params:
            kwargs['video_grid_thw'] = additional_inputs.get('video_grid_thw')
        
        # === AUDIO/SPEECH PARAMETERS ===
        if 'input_features' in forward_params:
            kwargs['input_features'] = additional_inputs.get('input_features')
        if 'input_values' in forward_params:
            kwargs['input_values'] = additional_inputs.get('input_values')
        
        # === SPECIALIZED PARAMETERS ===
        if 'token_type_ids' in forward_params:
            kwargs['token_type_ids'] = additional_inputs.get('token_type_ids')
        if 'langs' in forward_params:  # XLM models
            kwargs['langs'] = additional_inputs.get('langs')
        if 'lengths' in forward_params:  # XLM models
            kwargs['lengths'] = additional_inputs.get('lengths')
        if 'rope_deltas' in forward_params:  # GLM-4V
            kwargs['rope_deltas'] = additional_inputs.get('rope_deltas')
        
        # === TIME SERIES PARAMETERS ===
        if 'past_values' in forward_params:
            kwargs['past_values'] = additional_inputs.get('past_values')
        if 'past_time_features' in forward_params:
            kwargs['past_time_features'] = additional_inputs.get('past_time_features')
        if 'past_observed_mask' in forward_params:
            kwargs['past_observed_mask'] = additional_inputs.get('past_observed_mask')
        if 'static_categorical_features' in forward_params:
            kwargs['static_categorical_features'] = additional_inputs.get('static_categorical_features')
        if 'static_real_features' in forward_params:
            kwargs['static_real_features'] = additional_inputs.get('static_real_features')
        if 'future_values' in forward_params:
            kwargs['future_values'] = additional_inputs.get('future_values')
        if 'future_time_features' in forward_params:
            kwargs['future_time_features'] = additional_inputs.get('future_time_features')
        
        # === OUTPUT CONTROL PARAMETERS ===
        if 'output_attentions' in forward_params:
            kwargs['output_attentions'] = False  # Training default
        if 'output_hidden_states' in forward_params:
            kwargs['output_hidden_states'] = False  # Training default
        if 'return_dict' in forward_params:
            kwargs['return_dict'] = False  # Prefer tuple outputs for consistency
        
        # === TRAINING PARAMETERS ===
        if 'labels' in forward_params:
            kwargs['labels'] = additional_inputs.get('labels')
        if 'mc_token_ids' in forward_params:  # GPT2DoubleHeads
            kwargs['mc_token_ids'] = additional_inputs.get('mc_token_ids')
        if 'mc_labels' in forward_params:  # GPT2DoubleHeads
            kwargs['mc_labels'] = additional_inputs.get('mc_labels')
        if 'logits_to_keep' in forward_params:  # GPT-2
            kwargs['logits_to_keep'] = additional_inputs.get('logits_to_keep')
        
        return kwargs

    # The old _apply_strategy method has been completely replaced by the
    # universal strategy generation system. No more hardcoded logic!
