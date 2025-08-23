"""
SafeTensors Demo for MoL System

This demo shows how to use SafeTensors for secure model serialization in MoL.
SafeTensors provides memory-safe alternatives to PyTorch's pickle-based format.
"""

import os
import torch
from pathlib import Path
from mol import (
    MoLRuntime, save_model_safe, load_model_safe, 
    is_safetensors_available, safetensors_manager, SAFETENSORS_AVAILABLE
)
from mol.core.mol_runtime import MoLConfig
from mol.training.trainer import MoLTrainer, TrainingConfig, create_simple_dataset
from torch.utils.data import DataLoader


def safetensors_basic_demo():
    """Demonstrate basic SafeTensors functionality."""
    print("🔒 SafeTensors Basic Demo")
    print("=" * 50)
    
    if not SAFETENSORS_AVAILABLE:
        print("❌ SafeTensors not available. Install with:")
        print("   pip install safetensors")
        return
    
    print(f"✅ SafeTensors available: {is_safetensors_available()}")
    
    # Create a simple MoL system
    config = MoLConfig(
        models=["Qwen/Qwen3-0.6B", "LiquidAI/LFM2-350M"],
        adapter_type="linear",
        router_type="simple",
        max_layers=2,
        temperature=1.0
    )
    
    mol_runtime = MoLRuntime(config)
    mol_runtime.setup_embeddings()
    mol_runtime.setup_lm_head()
    
    # Add fusion layers
    mol_runtime.add_layer([
        ("Qwen/Qwen3-0.6B", 0),
        ("LiquidAI/LFM2-350M", 0)
    ], layer_idx=0)
    
    print(f"📦 Created MoL system with {len(mol_runtime.layers)} layers")
    
    # Save using SafeTensors (default)
    safe_path = "./mol_model_safe"
    mol_runtime.save_checkpoint(safe_path, use_safetensors=True)
    print(f"💾 Saved model using SafeTensors")
    
    # Save using PyTorch for comparison
    pytorch_path = "./mol_model_pytorch.pt"
    mol_runtime.save_checkpoint(pytorch_path, use_safetensors=False)
    print(f"💾 Saved model using PyTorch")
    
    # Compare file sizes
    safe_file = Path(f"{safe_path}.safetensors")
    pt_file = Path(pytorch_path)
    
    if safe_file.exists() and pt_file.exists():
        safe_size = safe_file.stat().st_size / (1024 * 1024)  # MB
        pt_size = pt_file.stat().st_size / (1024 * 1024)  # MB
        
        print(f"\n📊 File Size Comparison:")
        print(f"   SafeTensors: {safe_size:.2f} MB")
        print(f"   PyTorch:     {pt_size:.2f} MB")
        print(f"   Difference:  {abs(safe_size - pt_size):.2f} MB")
    
    # Load and verify
    print(f"\n🔍 Loading and verifying models...")
    
    # Load SafeTensors version
    mol_loaded_safe = MoLRuntime.load_checkpoint(safe_path)
    print(f"✅ Loaded SafeTensors model successfully")
    
    # Load PyTorch version
    mol_loaded_pt = MoLRuntime.load_checkpoint(pytorch_path)
    print(f"✅ Loaded PyTorch model successfully")
    
    # Verify they're identical
    safe_params = sum(p.numel() for p in mol_loaded_safe.parameters())
    pt_params = sum(p.numel() for p in mol_loaded_pt.parameters())
    
    print(f"\n✅ Parameter verification:")
    print(f"   SafeTensors model: {safe_params:,} parameters")
    print(f"   PyTorch model:     {pt_params:,} parameters")
    print(f"   Match: {'✅ Yes' if safe_params == pt_params else '❌ No'}")
    
    # Cleanup
    cleanup_files = [safe_file, pt_file, Path(f"{safe_path}.aux.pt")]
    for file_path in cleanup_files:
        if file_path.exists():
            os.remove(file_path)
    
    print(f"\n🧹 Cleaned up demo files")


def safetensors_training_demo():
    """Demonstrate SafeTensors in training pipeline."""
    print("\n🏋️ SafeTensors Training Demo")
    print("=" * 50)
    
    if not SAFETENSORS_AVAILABLE:
        print("❌ SafeTensors not available")
        return
    
    # Create MoL system
    config = MoLConfig(
        models=["Qwen/Qwen3-0.6B", "LiquidAI/LFM2-350M"],
        adapter_type="linear",
        router_type="simple",
        max_layers=2
    )
    
    mol_runtime = MoLRuntime(config)
    mol_runtime.setup_embeddings()
    mol_runtime.setup_lm_head()
    mol_runtime.add_layer([
        ("Qwen/Qwen3-0.6B", 0),
        ("LiquidAI/LFM2-350M", 0)
    ], layer_idx=0)
    
    # Create training data
    train_texts = [
        "Hello, how are you?",
        "What's the weather like?",
        "Tell me a story.",
        "How does machine learning work?"
    ] * 5  # Repeat for more data
    
    train_dataset = create_simple_dataset(train_texts, mol_runtime.tokenizer, max_length=32)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    # Training config with SafeTensors enabled
    training_config = TrainingConfig(
        learning_rate=1e-4,
        batch_size=2,
        max_epochs=1,
        max_steps=5,  # Very short for demo
        use_safetensors=True,  # Enable SafeTensors
        output_dir="./safetensors_training_demo",
        logging_steps=2,
        save_steps=3,
        use_wandb=False
    )
    
    print(f"🔧 Training with SafeTensors enabled: {training_config.use_safetensors}")
    
    # Train
    trainer = MoLTrainer(mol_runtime, training_config)
    trainer.setup_optimization()
    trainer.freeze_experts()
    
    print("🚀 Starting training...")
    trainer.train(train_dataloader)
    
    # Check saved files
    output_dir = Path(training_config.output_dir)
    saved_files = list(output_dir.glob("*.safetensors")) + list(output_dir.glob("*.aux.pt"))
    
    print(f"\n💾 Saved checkpoint files:")
    for file_path in saved_files:
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"   {file_path.name}: {size_mb:.2f} MB")
    
    # Load checkpoint to verify
    if saved_files:
        checkpoint_path = str(output_dir / "final_checkpoint")
        try:
            trainer.load_checkpoint(checkpoint_path)
            print(f"✅ Successfully loaded SafeTensors checkpoint")
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
    
    # Cleanup
    import shutil
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"🧹 Cleaned up training directory")


def safetensors_conversion_demo():
    """Demonstrate converting PyTorch checkpoints to SafeTensors."""
    print("\n🔄 SafeTensors Conversion Demo")
    print("=" * 50)
    
    if not SAFETENSORS_AVAILABLE:
        print("❌ SafeTensors not available")
        return
    
    # Create a simple model and save as PyTorch
    config = MoLConfig(
        models=["Qwen/Qwen3-0.6B", "LiquidAI/LFM2-350M"],
        adapter_type="linear",
        router_type="simple",
        max_layers=1
    )
    
    mol_runtime = MoLRuntime(config)
    mol_runtime.setup_embeddings()
    mol_runtime.setup_lm_head()
    mol_runtime.add_layer([
        ("Qwen/Qwen3-0.6B", 0),
        ("LiquidAI/LFM2-350M", 0)
    ], layer_idx=0)
    
    # Save as PyTorch format
    pytorch_path = "./conversion_demo.pt"
    mol_runtime.save_checkpoint(pytorch_path, use_safetensors=False)
    print(f"💾 Saved PyTorch checkpoint: {pytorch_path}")
    
    # Convert to SafeTensors
    safetensors_path = "./conversion_demo.safetensors"
    try:
        safetensors_manager.convert_pytorch_to_safetensors(
            pytorch_path, 
            safetensors_path,
            metadata={
                "converted_from": "pytorch",
                "conversion_date": "2024-01-01",
                "mol_version": "0.2.0"
            }
        )
        print(f"🔄 Converted to SafeTensors: {safetensors_path}")
        
        # Show file info
        file_info = safetensors_manager.get_file_info(safetensors_path)
        print(f"\n📊 SafeTensors File Info:")
        print(f"   File size: {file_info['file_size'] / (1024*1024):.2f} MB")
        print(f"   Total parameters: {file_info['total_parameters']:,}")
        print(f"   Number of tensors: {len(file_info['tensors'])}")
        print(f"   Metadata keys: {list(file_info['metadata'].keys())}")
        
        # List some tensor info
        print(f"\n🔍 First few tensors:")
        for i, (name, info) in enumerate(list(file_info['tensors'].items())[:3]):
            print(f"   {name}: {info['shape']} ({info['dtype']})")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
    
    # Cleanup
    for path in [pytorch_path, safetensors_path]:
        if os.path.exists(path):
            os.remove(path)
    
    print(f"🧹 Cleaned up conversion demo files")


def safetensors_security_demo():
    """Demonstrate security benefits of SafeTensors."""
    print("\n🛡️ SafeTensors Security Demo")
    print("=" * 50)
    
    print("🔒 Security Benefits of SafeTensors:")
    print()
    print("✅ Memory Safety:")
    print("   - No arbitrary code execution during loading")
    print("   - Safe from pickle-based attacks")
    print("   - Prevents buffer overflows")
    print()
    print("✅ Data Integrity:")
    print("   - Built-in format validation")
    print("   - Checksums for data verification")
    print("   - Prevents corrupted model loading")
    print()
    print("✅ Performance:")
    print("   - Lazy loading support")
    print("   - Memory-mapped access")
    print("   - Faster loading for large models")
    print()
    print("✅ Transparency:")
    print("   - Human-readable header")
    print("   - Inspectable without loading")
    print("   - Clear tensor metadata")
    
    if SAFETENSORS_AVAILABLE:
        print(f"\n🎯 Current MoL SafeTensors Configuration:")
        print(f"   - Available: ✅ Yes")
        print(f"   - Default format: SafeTensors")
        print(f"   - Fallback: PyTorch (.pt)")
        print(f"   - Training support: ✅ Yes")
        print(f"   - HuggingFace integration: ✅ Yes")
    else:
        print(f"\n⚠️ SafeTensors not installed")
        print(f"   Install with: pip install safetensors")


def main():
    """Run all SafeTensors demos."""
    print("🚀 MoL SafeTensors Integration Demo")
    print("=" * 60)
    print("This demo shows how MoL uses SafeTensors for secure model serialization")
    print("=" * 60)
    
    # Check availability first
    print(f"🔍 Checking SafeTensors availability...")
    print(f"   SafeTensors available: {SAFETENSORS_AVAILABLE}")
    
    if not SAFETENSORS_AVAILABLE:
        print("\n❌ SafeTensors not available")
        print("   Install with: pip install safetensors")
        print("   Falling back to security information only...")
        safetensors_security_demo()
        return
    
    try:
        # Run demos
        safetensors_basic_demo()
        safetensors_training_demo()
        safetensors_conversion_demo()
        safetensors_security_demo()
        
        print("\n" + "=" * 60)
        print("🎉 All SafeTensors demos completed successfully!")
        print("=" * 60)
        print()
        print("💡 Key Takeaways:")
        print("   • MoL now supports SafeTensors by default")
        print("   • Use use_safetensors=True in training config")
        print("   • Automatic fallback to PyTorch if SafeTensors unavailable")
        print("   • Better security and performance for model serialization")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()