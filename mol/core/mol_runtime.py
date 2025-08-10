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
        
        # Core components
        self.block_extractor = BlockExtractor()
        self.memory_manager = MemoryManager()
        self.model_utils = ModelUtils()
        
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
        self.tokenizer = AutoTokenizer.from_pretrained(primary_model)
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
        
        # Create router
        router = create_router(
            self.config.router_type,
            hidden_dim=self.target_hidden_dim,
            num_experts=len(experts),
            temperature=self.config.temperature,
            top_k=self.config.top_k_experts
        )
        
        # Create layer specification
        layer_spec = LayerSpec(
            experts=experts,
            adapters=adapters,
            router=router,
            layer_idx=layer_idx
        )
        
        # Wrap in MoL layer module
        mol_layer = MoLLayer(layer_spec, self.config)
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
        
        # Track router statistics
        router_stats = {} if return_router_stats else None
        
        # Embedding layer
        if self.embedding_layer is None:
            raise RuntimeError("Embedding layer not initialized. Call setup_embeddings() first.")
        
        hidden_states = self.embedding_layer(input_ids)
        
        # Apply embedding adapter if needed
        if self.embedding_adapter:
            hidden_states = self.embedding_adapter(hidden_states)
        
        # Process through MoL layers
        for layer_idx, mol_layer in enumerate(self.layers):
            hidden_states, layer_stats = mol_layer(
                hidden_states,
                attention_mask=attention_mask,
                return_stats=return_router_stats
            )
            
            if return_router_stats and layer_stats:
                router_stats[f"layer_{layer_idx}"] = layer_stats
        
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
    
    def save_checkpoint(self, path: str):
        """Save MoL checkpoint."""
        checkpoint = {
            'config': self.config,
            'target_hidden_dim': self.target_hidden_dim,
            'model_infos': self.model_infos,
            'state_dict': self.state_dict(),
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Saved MoL checkpoint to {path}")
    
    @classmethod
    def load_checkpoint(cls, path: str):
        """Load MoL checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        
        # Create runtime with loaded config
        runtime = cls(checkpoint['config'])
        runtime.target_hidden_dim = checkpoint['target_hidden_dim']
        runtime.model_infos = checkpoint['model_infos']
        
        # Load state dict
        runtime.load_state_dict(checkpoint['state_dict'])
        
        logger.info(f"Loaded MoL checkpoint from {path}")
        return runtime


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
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_stats: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Forward pass through MoL layer."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Get routing decisions
        expert_weights, router_logits = self.router(hidden_states, attention_mask)
        
        # Process through each expert
        expert_outputs = []
        for i, (expert, adapter) in enumerate(zip(self.experts, self.adapters)):
            # Pass through expert
            if hasattr(expert, 'forward'):
                expert_output = expert(hidden_states)
                # Handle different return formats
                if isinstance(expert_output, tuple):
                    expert_output = expert_output[0]  # Take hidden states
            else:
                expert_output = hidden_states  # Fallback
            
            # Apply adapter for dimension matching
            expert_output = adapter(expert_output)
            expert_outputs.append(expert_output)
        
        # Combine expert outputs using routing weights
        expert_outputs = torch.stack(expert_outputs, dim=-1)  # [batch, seq, hidden, num_experts]
        expert_weights = expert_weights.unsqueeze(-2)  # [batch, seq, 1, num_experts]
        
        # Weighted combination
        output = torch.sum(expert_outputs * expert_weights, dim=-1)  # [batch, seq, hidden]
        
        # Collect statistics
        stats = None
        if return_stats:
            stats = {
                'expert_weights': expert_weights.squeeze(-2).detach(),
                'router_entropy': compute_router_entropy(router_logits).item(),
                'load_balancing_loss': compute_load_balancing_loss(
                    expert_weights.squeeze(-2), attention_mask
                ).item(),
                'num_experts': self.num_experts,
            }
        
        return output, stats