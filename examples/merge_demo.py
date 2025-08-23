"""
Comprehensive demo showing MergeKit-style model merging capabilities.
"""

import os
import torch
from mol import SlerpMerge, TiesMerge, TaskArithmeticMerge, LinearMerge
from mol.config import ConfigParser, MergeConfiguration
from mol.merge_methods.base_merge import MergeConfig


def demo_slerp_merge():
    """Demonstrate SLERP merging."""
    print("🔄 SLERP Merge Demo")
    
    # Create configuration
    config = MergeConfig(
        method="slerp",
        models=["gpt2", "distilgpt2"],
        parameters={
            "t": 0.5,
            "eps": 1e-8
        },
        dtype="float16",
        device="cpu",
        output_path="./slerp_merged"
    )
    
    # Initialize merge method
    slerp = SlerpMerge(config)
    
    # Load models
    print("📥 Loading models...")
    models = slerp.load_models()
    
    # Perform merge
    print("🔀 Merging with SLERP...")
    merged_model = slerp.merge(models)
    
    print("✅ SLERP merge completed")
    return merged_model


def demo_yaml_config():
    """Demonstrate YAML configuration system."""
    print("📄 YAML Configuration Demo")
    
    # Create example YAML config
    yaml_config = """
merge_method: ties
slices:
  - model: gpt2
    layer_range: [0, 12] 
    weight: 1.0
  - model: distilgpt2
    layer_range: [0, 6]
    weight: 0.8
base_model: gpt2
parameters:
  density: 0.8
  normalize: true
  majority_sign_method: total
dtype: float16
output_path: ./ties_merged
"""
    
    # Save config to file
    with open("demo_config.yml", "w") as f:
        f.write(yaml_config)
    
    # Parse configuration
    parser = ConfigParser()
    config = parser.parse_config("demo_config.yml")
    
    print(f"✅ Parsed {config.merge_method} configuration")
    print(f"📋 Models: {config.get_model_list()}")
    
    # Clean up
    os.remove("demo_config.yml")
    
    return config


def demo_cli_usage():
    """Show CLI usage examples."""
    print("🖥️  CLI Usage Examples")
    
    examples = [
        "mol-merge config.yml ./merged_model",
        "mol-merge config.yml --device cuda --verbose",
        "mol-merge validate config.yml", 
        "mol-merge examples ./config_examples"
    ]
    
    print("💡 Example commands:")
    for example in examples:
        print(f"   {example}")


def main():
    """Run comprehensive demo."""
    print("🚀 MoL Enhanced Merge System Demo")
    print("=" * 50)
    
    try:
        # Demo SLERP merge
        merged_model = demo_slerp_merge()
        print()
        
        # Demo YAML config
        config = demo_yaml_config() 
        print()
        
        # Show CLI usage
        demo_cli_usage()
        print()
        
        print("✅ All demos completed successfully!")
        print("🎉 MoL system now has MergeKit-style capabilities!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()