"""
Model utility functions for MoL system.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from transformers import AutoConfig, AutoModel
import logging

logger = logging.getLogger(__name__)


class ModelUtils:
    """Utility functions for model operations."""
    
    @staticmethod
    def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
        """Count total parameters in a model."""
        if trainable_only:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in model.parameters())
    
    @staticmethod
    def freeze_parameters(module: nn.Module, freeze: bool = True):
        """Freeze or unfreeze module parameters."""
        for param in module.parameters():
            param.requires_grad = not freeze
        
        logger.info(f"{'Froze' if freeze else 'Unfroze'} {ModelUtils.count_parameters(module)} parameters")
    
    @staticmethod
    def get_layer_sizes(model: nn.Module) -> Dict[str, int]:
        """Get parameter count for each layer/module."""
        layer_sizes = {}
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                param_count = ModelUtils.count_parameters(module)
                if param_count > 0:
                    layer_sizes[name] = param_count
        return layer_sizes
    
    @staticmethod
    def print_model_info(model: nn.Module, name: str = "Model"):
        """Print detailed model information."""
        total_params = ModelUtils.count_parameters(model)
        trainable_params = ModelUtils.count_parameters(model, trainable_only=True)
        
        print(f"\n{name} Information:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")
        
        # Memory estimate
        memory_mb = total_params * 4 / (1024 * 1024)  # Assume float32
        print(f"Estimated memory: {memory_mb:.1f} MB")
        
        # Layer breakdown for smaller models
        if total_params < 1e6:  # Only for models < 1M parameters
            layer_sizes = ModelUtils.get_layer_sizes(model)
            print("\nLayer breakdown:")
            for name, size in sorted(layer_sizes.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {name}: {size:,} parameters")
    
    @staticmethod
    def init_weights_identity(layer: nn.Linear):
        """Initialize linear layer to approximate identity."""
        if layer.weight.shape[0] == layer.weight.shape[1]:
            # Perfect identity for square matrices
            nn.init.eye_(layer.weight)
        else:
            # Pseudo-identity for non-square matrices
            min_dim = min(layer.weight.shape)
            nn.init.zeros_(layer.weight)
            for i in range(min_dim):
                layer.weight[i, i] = 1.0
        
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    
    @staticmethod
    def init_weights_small_random(layer: nn.Linear, std: float = 0.01):
        """Initialize linear layer with small random weights."""
        nn.init.normal_(layer.weight, mean=0.0, std=std)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    
    @staticmethod
    def compute_activation_stats(
        activations: torch.Tensor
    ) -> Dict[str, float]:
        """Compute statistics for activation analysis."""
        with torch.no_grad():
            stats = {
                'mean': activations.mean().item(),
                'std': activations.std().item(),
                'min': activations.min().item(),
                'max': activations.max().item(),
                'l2_norm': activations.norm().item(),
                'sparsity': (activations.abs() < 1e-6).float().mean().item(),
            }
        return stats
    
    @staticmethod
    def compute_weight_similarity(
        weights1: torch.Tensor,
        weights2: torch.Tensor,
        method: str = "cosine"
    ) -> float:
        """Compute similarity between two weight tensors."""
        w1_flat = weights1.flatten()
        w2_flat = weights2.flatten()
        
        if method == "cosine":
            # Cosine similarity
            dot_product = torch.dot(w1_flat, w2_flat)
            norm1 = torch.norm(w1_flat)
            norm2 = torch.norm(w2_flat)
            similarity = (dot_product / (norm1 * norm2)).item()
        
        elif method == "l2":
            # Negative L2 distance (higher is more similar)
            l2_dist = torch.norm(w1_flat - w2_flat)
            similarity = -l2_dist.item()
        
        elif method == "pearson":
            # Pearson correlation
            mean1 = w1_flat.mean()
            mean2 = w2_flat.mean()
            centered1 = w1_flat - mean1
            centered2 = w2_flat - mean2
            
            numerator = torch.dot(centered1, centered2)
            denominator = torch.sqrt(torch.dot(centered1, centered1) * torch.dot(centered2, centered2))
            similarity = (numerator / denominator).item()
        
        else:
            raise ValueError(f"Unknown similarity method: {method}")
        
        return similarity
    
    @staticmethod
    def analyze_gradient_flow(model: nn.Module) -> Dict[str, float]:
        """Analyze gradient flow through the model."""
        gradient_stats = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradient_stats[name] = grad_norm
        
        if gradient_stats:
            total_norm = sum(gradient_stats.values())
            gradient_stats['total_norm'] = total_norm
        
        return gradient_stats
    
    @staticmethod
    def create_position_embeddings(
        max_length: int,
        hidden_dim: int,
        device: torch.device
    ) -> torch.Tensor:
        """Create sinusoidal position embeddings."""
        position = torch.arange(max_length, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float, device=device) *
            -(np.log(10000.0) / hidden_dim)
        )
        
        pos_embeddings = torch.zeros(max_length, hidden_dim, device=device)
        pos_embeddings[:, 0::2] = torch.sin(position * div_term)
        pos_embeddings[:, 1::2] = torch.cos(position * div_term)
        
        return pos_embeddings
    
    @staticmethod
    def interpolate_position_embeddings(
        pos_embeddings: torch.Tensor,
        old_length: int,
        new_length: int
    ) -> torch.Tensor:
        """Interpolate position embeddings to a new sequence length."""
        if old_length == new_length:
            return pos_embeddings
        
        # Use linear interpolation
        old_indices = torch.linspace(0, old_length - 1, old_length)
        new_indices = torch.linspace(0, old_length - 1, new_length)
        
        # Interpolate along the sequence dimension
        interpolated = torch.nn.functional.interpolate(
            pos_embeddings.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims
            size=new_length,
            mode='linear',
            align_corners=True
        )
        
        return interpolated.squeeze(0).squeeze(0)
    
    @staticmethod
    def merge_tokenizers(tokenizer_list: List[Any]) -> Dict[str, Any]:
        """
        Analyze tokenizer compatibility and suggest merge strategy.
        
        Returns information about vocab overlap and merge feasibility.
        """
        if len(tokenizer_list) == 0:
            return {}
        
        # Get vocabularies
        vocabs = []
        for tokenizer in tokenizer_list:
            if hasattr(tokenizer, 'get_vocab'):
                vocabs.append(set(tokenizer.get_vocab().keys()))
            else:
                vocabs.append(set())
        
        # Compute overlaps
        if len(vocabs) > 1:
            intersection = vocabs[0]
            union = vocabs[0]
            
            for vocab in vocabs[1:]:
                intersection = intersection.intersection(vocab)
                union = union.union(vocab)
            
            overlap_ratio = len(intersection) / len(union) if len(union) > 0 else 0.0
        else:
            overlap_ratio = 1.0
            intersection = vocabs[0] if vocabs else set()
            union = vocabs[0] if vocabs else set()
        
        merge_info = {
            'num_tokenizers': len(tokenizer_list),
            'vocab_sizes': [len(vocab) for vocab in vocabs],
            'intersection_size': len(intersection),
            'union_size': len(union),
            'overlap_ratio': overlap_ratio,
            'merge_feasible': overlap_ratio > 0.5,  # Arbitrary threshold
            'recommended_strategy': 'single' if overlap_ratio > 0.8 else 'mapping'
        }
        
        return merge_info
    
    @staticmethod
    def create_attention_mask(
        input_ids: torch.Tensor,
        pad_token_id: int
    ) -> torch.Tensor:
        """Create attention mask from input IDs."""
        return (input_ids != pad_token_id).long()
    
    @staticmethod
    def pad_sequences(
        sequences: List[torch.Tensor],
        pad_token_id: int = 0,
        max_length: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad sequences to the same length."""
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)
        
        batch_size = len(sequences)
        padded = torch.full(
            (batch_size, max_length),
            pad_token_id,
            dtype=sequences[0].dtype,
            device=sequences[0].device
        )
        
        attention_mask = torch.zeros(
            (batch_size, max_length),
            dtype=torch.long,
            device=sequences[0].device
        )
        
        for i, seq in enumerate(sequences):
            length = min(len(seq), max_length)
            padded[i, :length] = seq[:length]
            attention_mask[i, :length] = 1
        
        return padded, attention_mask
    
    @staticmethod
    def estimate_flops(
        model: nn.Module,
        input_shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32
    ) -> int:
        """Rough estimate of FLOPs for a forward pass."""
        # This is a very rough approximation
        total_params = ModelUtils.count_parameters(model)
        input_size = np.prod(input_shape)
        
        # Assume roughly 2 FLOPs per parameter per input element
        estimated_flops = total_params * input_size * 2
        
        return estimated_flops
    
    @staticmethod
    def profile_model_speed(
        model: nn.Module,
        input_tensor: torch.Tensor,
        num_runs: int = 10,
        warmup_runs: int = 3
    ) -> Dict[str, float]:
        """Profile model inference speed."""
        model.eval()
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_tensor)
        
        # Synchronize if using GPU
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Timed runs
        import time
        times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(input_tensor)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.time()
                times.append(end_time - start_time)
        
        times = np.array(times)
        
        return {
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'throughput': input_tensor.shape[0] / np.mean(times),  # samples/sec
        }