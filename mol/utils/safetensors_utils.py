"""
SafeTensors utilities for secure and efficient model serialization.
"""

import os
import json
import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from safetensors import safe_open
    from safetensors.torch import save_file as safe_save_file
    from safetensors.torch import load_file as safe_load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    safe_open = None
    safe_save_file = None
    safe_load_file = None
    SAFETENSORS_AVAILABLE = False
    logger.warning("SafeTensors not available. Install with: pip install safetensors")


class SafeTensorsManager:
    """
    Manager for SafeTensors operations in MoL system.
    
    Provides secure alternatives to PyTorch's pickle-based serialization.
    """
    
    def __init__(self):
        """Initialize SafeTensors manager."""
        if not SAFETENSORS_AVAILABLE:
            logger.warning("SafeTensors not available - falling back to PyTorch serialization")
    
    def save_model_state(
        self, 
        model: nn.Module, 
        save_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save model state dict using SafeTensors.
        
        Args:
            model: PyTorch model to save
            save_path: Path to save the model
            metadata: Optional metadata to include
        """
        save_path = Path(save_path)
        
        if not SAFETENSORS_AVAILABLE:
            logger.warning("SafeTensors not available, using PyTorch save")
            torch.save(model.state_dict(), save_path.with_suffix('.pt'))
            return
        
        try:
            # Get state dict
            state_dict = model.state_dict()
            
            # Convert metadata to strings for SafeTensors
            safe_metadata = {}
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        safe_metadata[key] = str(value)
                    else:
                        safe_metadata[key] = json.dumps(value)
            
            # Save with SafeTensors
            safetensors_path = save_path.with_suffix('.safetensors')
            safe_save_file(state_dict, safetensors_path, metadata=safe_metadata)
            
            logger.info(f"Saved model state to {safetensors_path} using SafeTensors")
            
        except Exception as e:
            logger.error(f"Failed to save with SafeTensors: {e}")
            logger.info("Falling back to PyTorch save")
            torch.save(model.state_dict(), save_path.with_suffix('.pt'))
    
    def load_model_state(
        self, 
        model: nn.Module, 
        load_path: Union[str, Path],
        device: str = "cpu",
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Load model state dict using SafeTensors.
        
        Args:
            model: PyTorch model to load state into
            load_path: Path to load the model from
            device: Device to load tensors to
            strict: Whether to strictly enforce state dict keys
            
        Returns:
            Metadata dictionary
        """
        load_path = Path(load_path)
        
        # Try SafeTensors first
        safetensors_path = load_path.with_suffix('.safetensors')
        if safetensors_path.exists() and SAFETENSORS_AVAILABLE:
            try:
                logger.info(f"Loading model state from {safetensors_path} using SafeTensors")
                
                # Load state dict
                state_dict = safe_load_file(safetensors_path, device=device)
                model.load_state_dict(state_dict, strict=strict)
                
                # Load metadata
                metadata = {}
                with safe_open(safetensors_path, framework="pt", device=device) as f:
                    metadata = f.metadata() or {}
                
                # Parse JSON metadata back
                parsed_metadata = {}
                for key, value in metadata.items():
                    try:
                        parsed_metadata[key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        parsed_metadata[key] = value
                
                return parsed_metadata
                
            except Exception as e:
                logger.warning(f"Failed to load with SafeTensors: {e}")
        
        # Fallback to PyTorch
        pt_path = load_path.with_suffix('.pt')
        if pt_path.exists():
            logger.info(f"Loading model state from {pt_path} using PyTorch")
            state_dict = torch.load(pt_path, map_location=device)
            model.load_state_dict(state_dict, strict=strict)
            return {}
        
        raise FileNotFoundError(f"No model file found at {load_path} (.safetensors or .pt)")
    
    def save_checkpoint(
        self,
        checkpoint_data: Dict[str, Any],
        save_path: Union[str, Path],
        use_safetensors: bool = True
    ) -> None:
        """
        Save training checkpoint with optional SafeTensors support.
        
        Args:
            checkpoint_data: Dictionary containing checkpoint data
            save_path: Path to save the checkpoint
            use_safetensors: Whether to use SafeTensors for model weights
        """
        save_path = Path(save_path)
        
        if use_safetensors and SAFETENSORS_AVAILABLE:
            try:
                # Separate model state dict from other data
                model_state = checkpoint_data.get('model_state_dict', {})
                other_data = {k: v for k, v in checkpoint_data.items() if k != 'model_state_dict'}
                
                # Save model weights with SafeTensors
                if model_state:
                    safetensors_path = save_path.with_suffix('.safetensors')
                    
                    # Convert other data to metadata strings
                    metadata = {}
                    for key, value in other_data.items():
                        if isinstance(value, (str, int, float, bool)):
                            metadata[f"checkpoint_{key}"] = str(value)
                        else:
                            try:
                                metadata[f"checkpoint_{key}"] = json.dumps(value, default=str)
                            except (TypeError, ValueError):
                                logger.warning(f"Could not serialize {key} to metadata")
                    
                    safe_save_file(model_state, safetensors_path, metadata=metadata)
                    logger.info(f"Saved checkpoint to {safetensors_path} using SafeTensors")
                    
                    # Save optimizer and scheduler states separately (they can't go in SafeTensors)
                    aux_data = {
                        k: v for k, v in other_data.items() 
                        if k in ['optimizer_state_dict', 'scheduler_state_dict']
                    }
                    if aux_data:
                        aux_path = save_path.with_suffix('.aux.pt')
                        torch.save(aux_data, aux_path)
                        logger.info(f"Saved auxiliary data to {aux_path}")
                    
                    return
                    
            except Exception as e:
                logger.warning(f"Failed to save checkpoint with SafeTensors: {e}")
        
        # Fallback to PyTorch
        pt_path = save_path.with_suffix('.pt')
        torch.save(checkpoint_data, pt_path)
        logger.info(f"Saved checkpoint to {pt_path} using PyTorch")
    
    def load_checkpoint(
        self,
        load_path: Union[str, Path],
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """
        Load training checkpoint with SafeTensors support.
        
        Args:
            load_path: Path to load the checkpoint from
            device: Device to load tensors to
            
        Returns:
            Checkpoint data dictionary
        """
        load_path = Path(load_path)
        
        # Try SafeTensors first
        safetensors_path = load_path.with_suffix('.safetensors')
        if safetensors_path.exists() and SAFETENSORS_AVAILABLE:
            try:
                logger.info(f"Loading checkpoint from {safetensors_path} using SafeTensors")
                
                # Load model state dict
                model_state_dict = safe_load_file(safetensors_path, device=device)
                
                # Load metadata
                metadata = {}
                with safe_open(safetensors_path, framework="pt", device=device) as f:
                    raw_metadata = f.metadata() or {}
                
                # Parse checkpoint metadata
                checkpoint_data = {'model_state_dict': model_state_dict}
                for key, value in raw_metadata.items():
                    if key.startswith('checkpoint_'):
                        real_key = key[11:]  # Remove 'checkpoint_' prefix
                        try:
                            checkpoint_data[real_key] = json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            checkpoint_data[real_key] = value
                
                # Load auxiliary data if exists
                aux_path = load_path.with_suffix('.aux.pt')
                if aux_path.exists():
                    aux_data = torch.load(aux_path, map_location=device)
                    checkpoint_data.update(aux_data)
                    logger.info(f"Loaded auxiliary data from {aux_path}")
                
                return checkpoint_data
                
            except Exception as e:
                logger.warning(f"Failed to load checkpoint with SafeTensors: {e}")
        
        # Fallback to PyTorch
        pt_path = load_path.with_suffix('.pt')
        if pt_path.exists():
            logger.info(f"Loading checkpoint from {pt_path} using PyTorch")
            return torch.load(pt_path, map_location=device)
        
        raise FileNotFoundError(f"No checkpoint file found at {load_path} (.safetensors or .pt)")
    
    def convert_pytorch_to_safetensors(
        self,
        pytorch_path: Union[str, Path],
        safetensors_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Convert existing PyTorch checkpoint to SafeTensors format.
        
        Args:
            pytorch_path: Path to PyTorch checkpoint
            safetensors_path: Path to save SafeTensors file
            metadata: Optional metadata to include
        """
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("SafeTensors not available")
        
        logger.info(f"Converting {pytorch_path} to SafeTensors format")
        
        # Load PyTorch checkpoint
        checkpoint = torch.load(pytorch_path, map_location='cpu')
        
        # Extract model state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # Convert other data to metadata
            safe_metadata = {}
            for key, value in checkpoint.items():
                if key != 'model_state_dict':
                    try:
                        safe_metadata[f"checkpoint_{key}"] = json.dumps(value, default=str)
                    except (TypeError, ValueError):
                        logger.warning(f"Could not convert {key} to metadata")
        else:
            # Assume it's just a state dict
            state_dict = checkpoint
            safe_metadata = {}
        
        # Add custom metadata
        if metadata:
            for key, value in metadata.items():
                safe_metadata[key] = str(value)
        
        # Save as SafeTensors
        safe_save_file(state_dict, safetensors_path, metadata=safe_metadata)
        logger.info(f"Converted to SafeTensors: {safetensors_path}")
    
    def list_tensors(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        List tensors in a SafeTensors file without loading them.
        
        Args:
            file_path: Path to SafeTensors file
            
        Returns:
            Dictionary with tensor names and their metadata
        """
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("SafeTensors not available")
        
        file_path = Path(file_path)
        
        tensor_info = {}
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                tensor_info[key] = {
                    'shape': list(tensor.shape),
                    'dtype': str(tensor.dtype),
                    'device': str(tensor.device)
                }
        
        return tensor_info
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get comprehensive information about a SafeTensors file.
        
        Args:
            file_path: Path to SafeTensors file
            
        Returns:
            File information dictionary
        """
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("SafeTensors not available")
        
        file_path = Path(file_path)
        
        info = {
            'file_size': file_path.stat().st_size,
            'tensors': self.list_tensors(file_path),
            'metadata': {}
        }
        
        # Get metadata
        with safe_open(file_path, framework="pt", device="cpu") as f:
            info['metadata'] = f.metadata() or {}
        
        # Calculate total parameters
        total_params = sum(
            torch.prod(torch.tensor(tensor_info['shape'])).item() 
            for tensor_info in info['tensors'].values()
        )
        info['total_parameters'] = total_params
        
        return info


# Global instance
safetensors_manager = SafeTensorsManager()


def is_safetensors_available() -> bool:
    """Check if SafeTensors is available."""
    return SAFETENSORS_AVAILABLE


def save_model_safe(
    model: nn.Module, 
    save_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Convenience function to save model using SafeTensors.
    
    Args:
        model: PyTorch model to save
        save_path: Path to save the model
        metadata: Optional metadata to include
    """
    safetensors_manager.save_model_state(model, save_path, metadata)


def load_model_safe(
    model: nn.Module, 
    load_path: Union[str, Path],
    device: str = "cpu",
    strict: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to load model using SafeTensors.
    
    Args:
        model: PyTorch model to load state into
        load_path: Path to load the model from
        device: Device to load tensors to
        strict: Whether to strictly enforce state dict keys
        
    Returns:
        Metadata dictionary
    """
    return safetensors_manager.load_model_state(model, load_path, device, strict)