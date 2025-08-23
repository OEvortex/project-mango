#!/usr/bin/env python3
"""
Debug router functionality directly.
"""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from mol.core.routers import SimpleRouter, create_router

def test_router():
    """Test router creation and forward pass."""
    print("Testing router functionality...")
    
    # Test SimpleRouter directly
    print("Creating SimpleRouter...")
    router = SimpleRouter(
        hidden_dim=512,
        num_experts=2,
        pooling_type="mean",
        temperature=1.0
    )
    
    print(f"Router created: {type(router)}")
    print(f"Router num_experts: {router.num_experts}")
    print(f"Router hidden_dim: {router.hidden_dim}")
    
    # Test input
    batch_size, seq_len, hidden_dim = 2, 10, 512
    x = torch.randn(batch_size, seq_len, hidden_dim)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    print(f"Input shape: {x.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    
    # Test forward pass
    print("Running forward pass...")
    try:
        result = router(x, attention_mask)
        print(f"Forward result type: {type(result)}")
        
        if result is None:
            print("ERROR: Router returned None!")
            return False
        
        if not isinstance(result, tuple):
            print(f"ERROR: Router returned {type(result)}, expected tuple!")
            return False
        
        if len(result) != 2:
            print(f"ERROR: Router returned tuple of length {len(result)}, expected 2!")
            return False
        
        expert_weights, router_logits = result
        print(f"Expert weights shape: {expert_weights.shape}")
        print(f"Router logits shape: {router_logits.shape}")
        
        # Validate shapes
        expected_weights_shape = (batch_size, seq_len, router.num_experts)
        expected_logits_shape = (batch_size, router.num_experts)
        
        if expert_weights.shape != expected_weights_shape:
            print(f"ERROR: Expert weights shape {expert_weights.shape}, expected {expected_weights_shape}")
            return False
        
        if router_logits.shape != expected_logits_shape:
            print(f"ERROR: Router logits shape {router_logits.shape}, expected {expected_logits_shape}")
            return False
        
        print("✓ Router test passed!")
        return True
        
    except Exception as e:
        print(f"ERROR: Router forward failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_create_router():
    """Test router factory function."""
    print("\nTesting router factory...")
    
    try:
        router = create_router(
            "simple",
            hidden_dim=512,
            num_experts=2,
            temperature=1.0
        )
        print(f"Factory created router: {type(router)}")
        
        if router is None:
            print("ERROR: Factory returned None!")
            return False
        
        print("✓ Router factory test passed!")
        return True
        
    except Exception as e:
        print(f"ERROR: Router factory failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 Router Debug Test")
    print("=" * 50)
    
    success = True
    success &= test_create_router()
    success &= test_router()
    
    if success:
        print("\n✅ All router tests passed!")
    else:
        print("\n❌ Router tests failed!")
        sys.exit(1)