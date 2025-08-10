"""
Router implementations for expert selection in MoL system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List
import math


class BaseRouter(nn.Module, ABC):
    """Base class for all routers."""
    
    def __init__(self, hidden_dim: int, num_experts: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
    
    @abstractmethod
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through router.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            attention_mask: Optional attention mask [batch_size, seq_len]
        
        Returns:
            expert_weights: Expert selection weights [batch_size, seq_len, num_experts] 
            router_logits: Raw router logits for entropy calculation
        """
        pass


class SimpleRouter(BaseRouter):
    """
    Simple pooled router that uses sequence-level pooling.
    
    Computes a single routing decision per sequence by pooling
    over the sequence dimension, then applies to all tokens.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        pooling_type: str = "mean",
        temperature: float = 1.0,
        dropout_rate: float = 0.1
    ):
        super().__init__(hidden_dim, num_experts)
        
        self.pooling_type = pooling_type
        self.temperature = temperature
        
        # Pooling layer
        if pooling_type not in ["mean", "max", "attention"]:
            raise ValueError(f"Unsupported pooling type: {pooling_type}")
        
        # Attention-based pooling
        if pooling_type == "attention":
            self.attention_pool = nn.Linear(hidden_dim, 1)
        
        # Router MLP
        self.router_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_experts),
        )
        
        # Initialize router weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize router weights for balanced selection."""
        for module in self.router_mlp:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _pool_sequence(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Pool sequence to single representation."""
        if self.pooling_type == "mean":
            if attention_mask is not None:
                # Masked mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(x)
                sum_embeddings = torch.sum(x * mask_expanded, dim=1)
                sum_mask = torch.sum(mask_expanded, dim=1)
                pooled = sum_embeddings / (sum_mask + 1e-9)
            else:
                pooled = torch.mean(x, dim=1)
        
        elif self.pooling_type == "max":
            if attention_mask is not None:
                # Set masked positions to very negative values
                x = x.masked_fill(~attention_mask.unsqueeze(-1), -1e9)
            pooled = torch.max(x, dim=1)[0]
        
        elif self.pooling_type == "attention":
            # Learned attention pooling
            attention_scores = self.attention_pool(x).squeeze(-1)  # [batch, seq_len]
            
            if attention_mask is not None:
                attention_scores = attention_scores.masked_fill(~attention_mask, -1e9)
            
            attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(-1)
            pooled = torch.sum(x * attention_weights, dim=1)
        
        return pooled
    
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through simple router."""
        batch_size, seq_len, hidden_dim = x.shape
        
        # Pool sequence to single representation
        pooled = self._pool_sequence(x, attention_mask)  # [batch_size, hidden_dim]
        
        # Get router logits
        router_logits = self.router_mlp(pooled)  # [batch_size, num_experts]
        
        # Apply temperature scaling
        router_logits = router_logits / self.temperature
        
        # Compute expert weights
        expert_weights = F.softmax(router_logits, dim=-1)  # [batch_size, num_experts]
        
        # Expand to match sequence length
        expert_weights = expert_weights.unsqueeze(1).expand(batch_size, seq_len, self.num_experts)
        
        # Return weights and logits for entropy calculation
        return expert_weights, router_logits


class TokenLevelRouter(BaseRouter):
    """
    Token-level router that makes routing decisions per token.
    
    More expressive than pooled routing but requires more computation.
    Supports top-k sparse routing for efficiency.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        dropout_rate: float = 0.1,
        noise_std: float = 0.1
    ):
        super().__init__(hidden_dim, num_experts)
        
        self.temperature = temperature
        self.top_k = top_k if top_k is not None else num_experts
        self.noise_std = noise_std
        
        # Router network
        self.router_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_experts),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize router weights for balanced selection."""
        for module in self.router_mlp:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _add_noise(self, logits: torch.Tensor) -> torch.Tensor:
        """Add noise to router logits during training for exploration."""
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise
        return logits
    
    def _top_k_gating(
        self, 
        logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply top-k sparse gating."""
        if self.top_k >= self.num_experts:
            # No sparsity needed
            weights = F.softmax(logits / self.temperature, dim=-1)
            return weights, logits
        
        # Get top-k experts
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        
        # Create sparse weights
        weights = torch.zeros_like(logits)
        top_k_weights = F.softmax(top_k_logits / self.temperature, dim=-1)
        
        # Scatter the weights back to full size
        weights.scatter_(-1, top_k_indices, top_k_weights)
        
        return weights, logits
    
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through token-level router."""
        batch_size, seq_len, hidden_dim = x.shape
        
        # Flatten for processing
        x_flat = x.view(-1, hidden_dim)  # [batch_size * seq_len, hidden_dim]
        
        # Get router logits for each token
        router_logits = self.router_mlp(x_flat)  # [batch_size * seq_len, num_experts]
        
        # Add noise during training
        router_logits = self._add_noise(router_logits)
        
        # Apply top-k gating
        expert_weights, _ = self._top_k_gating(router_logits)
        
        # Reshape back to sequence format
        expert_weights = expert_weights.view(batch_size, seq_len, self.num_experts)
        router_logits = router_logits.view(batch_size, seq_len, self.num_experts)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Zero out weights for masked tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(expert_weights)
            expert_weights = expert_weights * mask_expanded
        
        return expert_weights, router_logits


def create_router(
    router_type: str,
    hidden_dim: int,
    num_experts: int,
    **kwargs
) -> BaseRouter:
    """Factory function to create routers."""
    router_type = router_type.lower()
    
    if router_type == "simple":
        return SimpleRouter(hidden_dim, num_experts, **kwargs)
    elif router_type in ["token", "token_level"]:
        return TokenLevelRouter(hidden_dim, num_experts, **kwargs)
    else:
        raise ValueError(f"Unknown router type: {router_type}")


def compute_router_entropy(router_logits: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of router decisions for regularization.
    
    Higher entropy indicates more balanced expert usage.
    """
    probs = F.softmax(router_logits, dim=-1)
    log_probs = F.log_softmax(router_logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy.mean()


def compute_load_balancing_loss(
    expert_weights: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute load balancing loss to encourage even expert usage.
    
    Penalizes uneven distribution of load across experts.
    """
    if attention_mask is not None:
        # Only consider non-masked tokens
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(expert_weights)
        masked_weights = expert_weights * mask_expanded
        expert_usage = masked_weights.sum(dim=(0, 1))  # [num_experts]
        total_tokens = attention_mask.sum()
    else:
        expert_usage = expert_weights.sum(dim=(0, 1))  # [num_experts]
        total_tokens = expert_weights.shape[0] * expert_weights.shape[1]
    
    # Normalize by total tokens
    expert_usage = expert_usage / total_tokens
    
    # Compute coefficient of variation (std/mean) as load balancing metric
    mean_usage = expert_usage.mean()
    std_usage = expert_usage.std()
    load_balance_loss = std_usage / (mean_usage + 1e-8)
    
    return load_balance_loss