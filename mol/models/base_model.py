"""
Base model interface for MoL system.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn


class BaseMoLModel(ABC):
    """
    Abstract base class for models used in MoL system.
    
    Provides a unified interface for different model architectures.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.config = None
        self.tokenizer = None
    
    @abstractmethod
    def load_model(self, device: str = "cpu", **kwargs) -> nn.Module:
        """Load the model."""
        pass
    
    @abstractmethod
    def get_layers(self) -> nn.ModuleList:
        """Get the transformer layers from the model."""
        pass
    
    @abstractmethod
    def get_embeddings(self) -> nn.Module:
        """Get the embedding layer."""
        pass
    
    @abstractmethod
    def get_lm_head(self) -> Optional[nn.Module]:
        """Get the language modeling head if available."""
        pass
    
    @abstractmethod
    def get_hidden_dim(self) -> int:
        """Get the hidden dimension of the model."""
        pass
    
    @abstractmethod
    def get_num_layers(self) -> int:
        """Get the number of transformer layers."""
        pass
    
    @abstractmethod
    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get general model information."""
        return {
            "model_name": self.model_name,
            "hidden_dim": self.get_hidden_dim(),
            "num_layers": self.get_num_layers(),
            "vocab_size": self.get_vocab_size(),
            "device": next(self.model.parameters()).device if self.model else None,
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"