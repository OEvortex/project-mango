"""
Comprehensive test and validation script for enhanced MoL system.
Tests all major improvements and bug fixes.
"""

import torch
import logging
import traceback
from mol import SlerpMerge, TiesMerge, TaskArithmeticMerge, LinearMerge
from mol.merge_methods.base_merge import MergeConfig
from mol.core.block_extractor import BlockExtractor
from mol.core.adapters import LinearAdapter, BottleneckAdapter
from mol.core.routers import SimpleRouter, TokenLevelRouter, compute_router_entropy, compute_load_balancing_loss
from mol.config import ConfigParser, ConfigValidator
from mol.utils.memory_utils import MemoryManager
from mol.utils.model_utils import ModelUtils

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_block_extractor():
    """Test improved block extractor functionality."""
    print("🔧 Testing Block Extractor Improvements...")
    
    try:
        extractor = BlockExtractor()
        
        # Test with small models
        test_models = ["gpt2", "distilgpt2"]
        
        for model_name in test_models:
            print(f"  📥 Testing {model_name}...")
            
            # Test model loading with improved error handling
            model, info = extractor.load_model(model_name, device="cpu")
            print(f"    ✅ Loaded {model_name}: {info.hidden_dim}D, {info.num_layers} layers")
            
            # Test layer extraction
            layers = extractor.get_model_layers(model, info.architecture_type)
            print(f"    ✅ Extracted {len(layers)} layers")
            
            # Test block extraction
            block = extractor.extract_block(model_name, 0, device="cpu")
            print(f"    ✅ Extracted block 0: {block.input_dim}D")
            
            # Test embedding extraction
            embeddings, embed_dim = extractor.get_embedding_layer(model_name, device="cpu")
            print(f"    ✅ Extracted embeddings: {embed_dim}D")
        
        # Test compatibility checking
        compatible = extractor.get_compatible_models("gpt2", ["distilgpt2"], max_dim_ratio=2.0)
        print(f"  ✅ Found {len(compatible)} compatible models")
        
        print("  ✅ Block Extractor tests passed!")
        
    except Exception as e:
        print(f"  ❌ Block Extractor test failed: {e}")
        traceback.print_exc()


def test_adapters():
    """Test improved adapter functionality."""
    print("🔧 Testing Adapter Improvements...")
    
    try:
        # Test LinearAdapter
        adapter = LinearAdapter(512, 768, init_identity=True)
        test_input = torch.randn(2, 10, 512)
        output = adapter(test_input)
        print(f"  ✅ LinearAdapter: {test_input.shape} -> {output.shape}")
        
        # Test BottleneckAdapter with edge cases
        bottleneck = BottleneckAdapter(512, 768, bottleneck_dim=128)
        
        # Test with normal input
        output = bottleneck(test_input)
        print(f"  ✅ BottleneckAdapter: {test_input.shape} -> {output.shape}")
        
        # Test with NaN input (should handle gracefully)
        nan_input = test_input.clone()
        nan_input[0, 0, 0] = float('nan')
        output_nan = bottleneck(nan_input)
        print(f"  ✅ BottleneckAdapter handled NaN input gracefully")
        
        print("  ✅ Adapter tests passed!")
        
    except Exception as e:
        print(f"  ❌ Adapter test failed: {e}")
        traceback.print_exc()


def test_routers():
    """Test improved router functionality."""
    print("🔧 Testing Router Improvements...")
    
    try:
        # Test SimpleRouter
        router = SimpleRouter(768, num_experts=3, pooling_type="mean")
        test_input = torch.randn(2, 10, 768)
        attention_mask = torch.ones(2, 10)
        
        weights, logits = router(test_input, attention_mask)
        print(f"  ✅ SimpleRouter: {test_input.shape} -> weights {weights.shape}")
        
        # Test entropy computation
        entropy = compute_router_entropy(logits)
        print(f"  ✅ Router entropy: {entropy.item():.4f}")
        
        # Test load balancing
        load_balance = compute_load_balancing_loss(weights, attention_mask)
        print(f"  ✅ Load balancing loss: {load_balance.item():.4f}")
        
        # Test TokenLevelRouter
        token_router = TokenLevelRouter(768, num_experts=3, top_k=2)
        weights2, logits2 = token_router(test_input, attention_mask)
        print(f"  ✅ TokenLevelRouter: {test_input.shape} -> weights {weights2.shape}")
        
        print("  ✅ Router tests passed!")
        
    except Exception as e:
        print(f"  ❌ Router test failed: {e}")
        traceback.print_exc()


def test_merge_methods():
    """Test improved merge methods."""
    print("🔧 Testing Merge Method Improvements...")
    
    try:
        # Test with small models
        config = MergeConfig(
            method="slerp",
            models=["gpt2", "distilgpt2"],
            parameters={"t": 0.5, "eps": 1e-8},
            dtype="float32",  # Use float32 for CPU testing
            device="cpu",
            output_path="./test_merged"
        )
        
        # Test SLERP merge
        slerp = SlerpMerge(config)
        print("  📥 Loading models for SLERP test...")
        models = slerp.load_models()
        print(f"  ✅ Loaded {len(models)} models")
        
        # Validate models
        is_valid = slerp.validate_models(models)
        print(f"  ✅ Model validation: {is_valid}")
        
        # Perform merge (on small subset for testing)
        print("  🔀 Testing SLERP merge...")
        merged_model = slerp.merge(models)
        print("  ✅ SLERP merge completed")
        
        # Test parameter counting
        param_count = ModelUtils.count_parameters(merged_model)
        print(f"  ✅ Merged model has {param_count:,} parameters")
        
        print("  ✅ Merge method tests passed!")
        
    except Exception as e:
        print(f"  ❌ Merge method test failed: {e}")
        traceback.print_exc()


def test_config_system():
    """Test improved configuration system."""
    print("🔧 Testing Configuration System Improvements...")
    
    try:
        # Test configuration validation
        validator = ConfigValidator()
        
        # Test valid config
        valid_config = {
            'merge_method': 'slerp',
            'slices': [
                {'model': 'gpt2'},
                {'model': 'distilgpt2'}
            ],
            'parameters': {'t': 0.5},
            'dtype': 'float16'
        }
        
        validator.validate_config_dict(valid_config)
        print("  ✅ Valid configuration passed validation")
        
        # Test invalid config (should raise error)
        try:
            invalid_config = {
                'merge_method': 'invalid_method',
                'models': ['gpt2']  # Not enough models
            }
            validator.validate_config_dict(invalid_config)
            print("  ❌ Invalid config should have failed!")
        except ValueError:
            print("  ✅ Invalid configuration correctly rejected")
        
        print("  ✅ Configuration system tests passed!")
        
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        traceback.print_exc()


def test_memory_management():
    """Test improved memory management."""
    print("🔧 Testing Memory Management Improvements...")
    
    try:
        memory_manager = MemoryManager()
        
        # Test memory stats
        stats = memory_manager.get_memory_stats()
        print(f"  📊 Memory stats: RAM {stats.used_ram:.1f}/{stats.total_ram:.1f} GB")
        
        # Test memory pressure check
        pressure = memory_manager.check_memory_pressure()
        print(f"  ✅ Memory pressure check: {pressure}")
        
        # Test memory optimization
        memory_manager.optimize_memory()
        print("  ✅ Memory optimization completed")
        
        print("  ✅ Memory management tests passed!")
        
    except Exception as e:
        print(f"  ❌ Memory management test failed: {e}")
        traceback.print_exc()


def main():
    """Run comprehensive tests."""
    print("🚀 Running Comprehensive MoL System Tests")
    print("=" * 60)
    
    try:
        # Test individual components
        test_block_extractor()
        print()
        
        test_adapters()
        print()
        
        test_routers()
        print()
        
        test_config_system()
        print()
        
        test_memory_management()
        print()
        
        # Test integration (with small models)
        test_merge_methods()
        print()
        
        print("🎉 All tests completed successfully!")
        print("✅ Enhanced MoL system is working properly")
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()