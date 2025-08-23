"""
Hugging Face Hub utilities for MoL system.
"""

import os
import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, Any, Union
from pathlib import Path
import json
import tempfile
import shutil

logger = logging.getLogger(__name__)

try:
    from huggingface_hub import HfApi, Repository, create_repo
    from transformers import AutoTokenizer, AutoConfig
    HF_HUB_AVAILABLE = True
except ImportError:
    HfApi = Repository = create_repo = None
    AutoTokenizer = AutoConfig = None
    HF_HUB_AVAILABLE = False
    logger.warning("Hugging Face Hub not available. Install with: pip install huggingface_hub")


class HuggingFacePublisher:
    """
    Publisher for pushing MoL models to Hugging Face Hub.
    
    Supports both lightweight MoL runtime and fully fused static models.
    """
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize HF publisher.
        
        Args:
            token: HuggingFace API token. If None, uses HF_TOKEN env var or login.
        """
        if not HF_HUB_AVAILABLE:
            raise ImportError(
                "Hugging Face Hub not available. Install with: pip install huggingface_hub transformers"
            )
        
        self.api = HfApi(token=token)
        self.token = token
    
    def push_mol_runtime(
        self,
        mol_runtime,
        repo_id: str,
        commit_message: str = "Upload MoL runtime model",
        private: bool = False,
        create_pr: bool = False
    ) -> str:
        """
        Push lightweight MoL runtime to Hugging Face.
        
        Args:
            mol_runtime: MoL runtime instance
            repo_id: Repository ID (username/model-name)
            commit_message: Commit message
            private: Whether to create private repository
            create_pr: Whether to create pull request
            
        Returns:
            Repository URL
        """
        logger.info(f"Pushing MoL runtime to {repo_id}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save MoL checkpoint
            checkpoint_path = temp_path / "mol_runtime.pt"
            mol_runtime.save_checkpoint(str(checkpoint_path))
            
            # Create model card
            self._create_mol_model_card(temp_path, mol_runtime, is_runtime=True)
            
            # Create config file
            self._create_mol_config(temp_path, mol_runtime)
            
            # Save tokenizer if available
            if mol_runtime.tokenizer:
                mol_runtime.tokenizer.save_pretrained(temp_path)
            
            # Create repository if it doesn't exist
            try:
                self.api.create_repo(repo_id, private=private, exist_ok=True)
            except Exception as e:
                logger.warning(f"Repository might already exist: {e}")
            
            # Upload files
            self.api.upload_folder(
                folder_path=temp_path,
                repo_id=repo_id,
                commit_message=commit_message,
                create_pr=create_pr
            )
        
        repo_url = f"https://huggingface.co/{repo_id}"
        logger.info(f"✅ MoL runtime uploaded to {repo_url}")
        return repo_url
    
    def push_fused_model(
        self,
        mol_runtime,
        repo_id: str,
        commit_message: str = "Upload fully fused model",
        private: bool = False,
        create_pr: bool = False,
        fusion_method: str = "weighted_average"
    ) -> str:
        """
        Create and push a fully fused static model to Hugging Face.
        
        Args:
            mol_runtime: MoL runtime instance
            repo_id: Repository ID (username/model-name)
            commit_message: Commit message
            private: Whether to create private repository
            create_pr: Whether to create pull request
            fusion_method: Method to fuse experts ('weighted_average', 'best_expert', 'learned_weights')
            
        Returns:
            Repository URL
        """
        logger.info(f"Creating and pushing fully fused model to {repo_id}")
        
        # Create fused model
        fused_model = self._create_fused_model(mol_runtime, fusion_method)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save fused model in HuggingFace format
            try:
                fused_model.save_pretrained(temp_path)
            except Exception as e:
                logger.warning(f"Could not use save_pretrained, using manual save: {e}")
                # Manual save with SafeTensors support
                from .safetensors_utils import save_model_safe, is_safetensors_available
                
                config_dict = self._extract_model_config(mol_runtime)
                
                # Save config
                config = AutoConfig.from_dict(config_dict)
                config.save_pretrained(temp_path)
                
                # Save model weights using SafeTensors if available
                if is_safetensors_available():
                    save_model_safe(fused_model, temp_path / "model", metadata={
                        "fusion_method": fusion_method,
                        "source_models": ",".join(mol_runtime.config.models)
                    })
                else:
                    torch.save(fused_model.state_dict(), temp_path / "pytorch_model.bin")
            
            # Save tokenizer
            if mol_runtime.tokenizer:
                mol_runtime.tokenizer.save_pretrained(temp_path)
            
            # Create model card
            self._create_mol_model_card(temp_path, mol_runtime, is_runtime=False, fusion_method=fusion_method)
            
            # Create repository if it doesn't exist
            try:
                self.api.create_repo(repo_id, private=private, exist_ok=True)
            except Exception as e:
                logger.warning(f"Repository might already exist: {e}")
            
            # Upload files
            self.api.upload_folder(
                folder_path=temp_path,
                repo_id=repo_id,
                commit_message=commit_message,
                create_pr=create_pr
            )
        
        repo_url = f"https://huggingface.co/{repo_id}"
        logger.info(f"✅ Fully fused model uploaded to {repo_url}")
        return repo_url
    
    def _create_fused_model(self, mol_runtime, fusion_method: str) -> nn.Module:
        """
        Create a fully fused static model from MoL runtime.
        
        Args:
            mol_runtime: MoL runtime instance
            fusion_method: Fusion strategy
            
        Returns:
            Fused PyTorch model
        """
        logger.info(f"Creating fused model using {fusion_method} method")
        
        if fusion_method == "weighted_average":
            return self._fuse_weighted_average(mol_runtime)
        elif fusion_method == "best_expert":
            return self._fuse_best_expert(mol_runtime)
        elif fusion_method == "learned_weights":
            return self._fuse_learned_weights(mol_runtime)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    def _fuse_weighted_average(self, mol_runtime) -> nn.Module:
        """Fuse experts using equal weighted average."""
        from ..core.block_extractor import BlockExtractor
        
        # Get primary model as base
        primary_model_name = mol_runtime.config.models[0]
        extractor = BlockExtractor()
        base_model, _ = extractor.load_model(primary_model_name, device="cpu")
        
        # Create fused model structure
        fused_model = type(base_model)(base_model.config)
        fused_model.load_state_dict(base_model.state_dict())
        
        # Fuse each layer
        for layer_idx, mol_layer in enumerate(mol_runtime.layers):
            if len(mol_layer.experts) <= 1:
                continue
                
            # Get corresponding layer in fused model
            fused_layer = self._get_model_layer(fused_model, layer_idx)
            if fused_layer is None:
                continue
            
            # Average expert weights
            expert_weights = []
            for expert in mol_layer.experts:
                expert_weights.append(expert.state_dict())
            
            # Compute averaged weights
            if expert_weights:
                averaged_state = {}
                for key in expert_weights[0].keys():
                    averaged_state[key] = torch.stack([
                        weights[key] for weights in expert_weights
                    ]).mean(dim=0)
                
                # Load averaged weights
                try:
                    fused_layer.load_state_dict(averaged_state, strict=False)
                except Exception as e:
                    logger.warning(f"Could not load averaged weights for layer {layer_idx}: {e}")
        
        logger.info("Created fused model using weighted average")
        return fused_model
    
    def _fuse_best_expert(self, mol_runtime) -> nn.Module:
        """Fuse by selecting the best performing expert per layer."""
        # For now, use the first expert (primary model)
        # In practice, this could be determined by validation performance
        primary_model_name = mol_runtime.config.models[0]
        extractor = BlockExtractor()
        base_model, _ = extractor.load_model(primary_model_name, device="cpu")
        
        logger.info("Created fused model using best expert (primary model)")
        return base_model
    
    def _fuse_learned_weights(self, mol_runtime) -> nn.Module:
        """Fuse using learned router weights as combination weights."""
        # Use router probabilities to weight expert combinations
        return self._fuse_weighted_average(mol_runtime)  # Simplified for now
    
    def _get_model_layer(self, model: nn.Module, layer_idx: int) -> Optional[nn.Module]:
        """Get specific transformer layer from model."""
        try:
            # Try common layer access patterns
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                return model.transformer.h[layer_idx]
            elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
                return model.model.layers[layer_idx]
            elif hasattr(model, 'bert') and hasattr(model.bert.encoder, 'layer'):
                return model.bert.encoder.layer[layer_idx]
            else:
                return None
        except (IndexError, AttributeError):
            return None
    
    def _extract_model_config(self, mol_runtime) -> Dict[str, Any]:
        """Extract model configuration for HuggingFace format."""
        # Get config from primary model
        primary_model_name = mol_runtime.config.models[0]
        
        try:
            config = AutoConfig.from_pretrained(primary_model_name)
            config_dict = config.to_dict()
        except Exception:
            # Fallback minimal config
            config_dict = {
                "model_type": "mol_fused",
                "hidden_size": mol_runtime.target_hidden_dim,
                "num_attention_heads": 12,
                "num_hidden_layers": len(mol_runtime.layers),
                "intermediate_size": mol_runtime.target_hidden_dim * 4,
                "vocab_size": 50257,  # GPT-2 default
            }
        
        # Add MoL-specific metadata
        config_dict.update({
            "mol_fused": True,
            "mol_source_models": mol_runtime.config.models,
            "mol_fusion_method": "static_fused",
        })
        
        return config_dict
    
    def _create_mol_config(self, save_path: Path, mol_runtime):
        """Create MoL configuration file."""
        mol_config = {
            "mol_runtime": True,
            "models": mol_runtime.config.models,
            "adapter_type": mol_runtime.config.adapter_type,
            "router_type": mol_runtime.config.router_type,
            "target_hidden_dim": mol_runtime.target_hidden_dim,
            "num_layers": len(mol_runtime.layers),
            "temperature": mol_runtime.config.temperature,
        }
        
        with open(save_path / "mol_config.json", "w") as f:
            json.dump(mol_config, f, indent=2)
    
    def _create_mol_model_card(
        self, 
        save_path: Path, 
        mol_runtime, 
        is_runtime: bool = True,
        fusion_method: str = None
    ):
        """Create README.md model card."""
        if is_runtime:
            title = "MoL Runtime Model"
            description = "This is a Modular Layer (MoL) runtime model that dynamically combines multiple LLMs."
            usage_note = "Requires MoL system to load and use."
        else:
            title = f"MoL Fused Model ({fusion_method})"
            description = f"This is a fully fused static model created from MoL runtime using {fusion_method} fusion."
            usage_note = "Can be used as a standard HuggingFace model."
        
        model_card = f"""---
license: apache-2.0
library_name: transformers
tags:
- mol
- model-fusion
- {'dynamic-routing' if is_runtime else 'static-fusion'}
---

# {title}

{description}

## Source Models
{chr(10).join(f"- {model}" for model in mol_runtime.config.models)}

## Model Details
- **Target Hidden Dimension**: {mol_runtime.target_hidden_dim}
- **Number of Layers**: {len(mol_runtime.layers)}
- **Adapter Type**: {mol_runtime.config.adapter_type}
- **Router Type**: {mol_runtime.config.router_type}
{'- **Fusion Method**: ' + fusion_method if fusion_method else ''}

## Usage

```python
{'from mol import MoLRuntime' if is_runtime else 'from transformers import AutoModel, AutoTokenizer'}

# Load model
{'model = MoLRuntime.load_checkpoint("mol_runtime.pt")' if is_runtime else 'model = AutoModel.from_pretrained("' + save_path.parent.name + '")'}
{'tokenizer = model.tokenizer' if is_runtime else 'tokenizer = AutoTokenizer.from_pretrained("' + save_path.parent.name + '")'}

# Use model for inference
inputs = tokenizer("Hello, world!", return_tensors="pt")
{'outputs = model.generate(**inputs)' if is_runtime else 'outputs = model(**inputs)'}
```

## Technical Details

{usage_note}

### Architecture
The model combines expertise from multiple source models using {'dynamic routing' if is_runtime else 'static fusion'}.

### Training
{'This runtime includes trained adapters and routers.' if is_runtime else 'This model represents a static fusion of the source models.'}

## Citation

```bibtex
@misc{{mol_system,
  title={{MoL: Modular Layer System for LLM Fusion}},
  author={{Project Mango Team}},
  year={{2024}},
  howpublished={{\\url{{https://github.com/project-mango/mol}}}}
}}
```
"""
        
        with open(save_path / "README.md", "w") as f:
            f.write(model_card)


def push_mol_to_hf(
    mol_runtime,
    repo_id: str,
    fusion_type: str = "runtime",
    token: Optional[str] = None,
    **kwargs
) -> str:
    """
    Convenience function to push MoL model to Hugging Face.
    
    Args:
        mol_runtime: MoL runtime instance
        repo_id: Repository ID (username/model-name)
        fusion_type: "runtime" for lightweight MoL or "fused" for full static model
        token: HuggingFace API token
        **kwargs: Additional arguments for push methods
        
    Returns:
        Repository URL
    """
    publisher = HuggingFacePublisher(token=token)
    
    if fusion_type == "runtime":
        return publisher.push_mol_runtime(mol_runtime, repo_id, **kwargs)
    elif fusion_type == "fused":
        return publisher.push_fused_model(mol_runtime, repo_id, **kwargs)
    else:
        raise ValueError("fusion_type must be 'runtime' or 'fused'")