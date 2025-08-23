#!/usr/bin/env python3
"""
Debug MoL layer creation and forward pass.
"""

import torch
import sys
import os
import logging
sys.path.insert(0, os.path.dirname(__file__))

# Enable debug logging only for MoL modules
logging.basicConfig(level=logging.WARNING)
mol_logger = logging.getLogger('mol')
mol_logger.setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

def test_mol_minimal():
    """Test MoL with minimal configuration."""
    print("Testing minimal MoL configuration...")
    
    try:
        from mol.core.mol_runtime import MoLRuntime, MoLConfig
        
        # Use very small, fast models
        mol_config = MoLConfig(
            models=['Qwen/Qwen3-0.6B', 'suayptalha/Qwen3-0.6B-Medical-Expert'],
            adapter_type="linear",
            router_type="simple",
            max_layers=1,  # Only 1 layer for quick test
            temperature=1.0,
            entropy_penalty_coeff=0.1,
            load_balancing_coeff=0.01,
            memory_efficient=True
        )
        
        print(f"Config: {mol_config.models}")
        
        # Initialize MoL runtime
        print("Creating MoL runtime...")
        mol_runtime = MoLRuntime(mol_config)
        
        print(f"Target hidden dim: {mol_runtime.target_hidden_dim}")
        print(f"Model infos: {list(mol_runtime.model_infos.keys())}")
        
        # Setup embeddings and LM head
        print("Setting up embeddings...")
        mol_runtime.setup_embeddings()
        
        print("Setting up LM head...")
        mol_runtime.setup_lm_head()
        
        # Add a single layer 
        print("Adding MoL layer...")
        mol_runtime.add_layer([
            ("Qwen/Qwen3-0.6B", 0),
            ("suayptalha/Qwen3-0.6B-Medical-Expert", 0)
        ], layer_idx=0)
        
        print(f"Added {len(mol_runtime.layers)} layers")
        
        # Test forward pass
        print("Testing forward pass...")
        inputs = mol_runtime.tokenizer("Hello world", return_tensors="pt", padding=True, truncation=True)
        
        print(f"Input IDs shape: {inputs['input_ids'].shape}")
        print(f"Attention mask shape: {inputs['attention_mask'].shape}")
        
        mol_runtime.eval()
        with torch.no_grad():
            try:
                output, stats = mol_runtime.forward(
                    inputs['input_ids'],
                    inputs['attention_mask'],
                    return_router_stats=True
                )
                print(f"Output shape: {output.shape}")
                print(f"Stats: {stats}")
                print("✅ Forward pass successful!")
                return True
                
            except Exception as e:
                print(f"❌ Forward pass failed: {e}")
                import traceback
                traceback.print_exc()
                return False
                
    except Exception as e:
        print(f"❌ MoL setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 MoL Minimal Test")
    print("=" * 50)
    
    success = test_mol_minimal()
    
    if success:
        print("\n✅ MoL minimal test passed!")
    else:
        print("\n❌ MoL minimal test failed!")
        sys.exit(1)