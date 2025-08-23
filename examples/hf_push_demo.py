"""
Example: Pushing MoL Models to Hugging Face Hub

This script demonstrates how to:
1. Create a MoL fusion model
2. Train it (optional)
3. Push both runtime and fully fused versions to HuggingFace
"""

import os
import torch
from mol import MoLRuntime, HuggingFacePublisher, push_mol_to_hf, HF_AVAILABLE
from mol.core.mol_runtime import MoLConfig
from mol.training.trainer import MoLTrainer, TrainingConfig, create_simple_dataset
from torch.utils.data import DataLoader


def create_and_push_mol_model():
    """Create a MoL model and push it to HuggingFace in different formats."""
    
    if not HF_AVAILABLE:
        print("❌ Hugging Face Hub not available. Install with:")
        print("   pip install huggingface_hub")
        return
    
    print("🚀 MoL to HuggingFace Demo")
    print("=" * 50)
    
    # 1. Create MoL Configuration
    config = MoLConfig(
        models=[
            "microsoft/DialoGPT-small",  # ~117M parameters
            "distilgpt2",                # ~82M parameters
        ],
        adapter_type="linear",
        router_type="simple",
        max_layers=4,
        temperature=1.0,
        memory_efficient=True
    )
    
    print(f"📋 Creating MoL system with models: {config.models}")
    
    # 2. Initialize MoL Runtime
    mol_runtime = MoLRuntime(config)
    mol_runtime.setup_embeddings()
    mol_runtime.setup_lm_head()
    
    # Add fusion layers
    print("🔧 Adding fusion layers...")
    mol_runtime.add_layer([
        ("microsoft/DialoGPT-small", 0),
        ("distilgpt2", 0)
    ], layer_idx=0)
    
    mol_runtime.add_layer([
        ("microsoft/DialoGPT-small", 1),
        ("distilgpt2", 1)
    ], layer_idx=1)
    
    print(f"✅ Created MoL system with {len(mol_runtime.layers)} fusion layers")
    
    # 3. Optional: Quick training (commented out for demo)
    # print("🏋️ Training adapters and routers...")
    # train_mol_system(mol_runtime)
    
    # 4. Save local checkpoint
    checkpoint_path = "./mol_demo_checkpoint.pt"
    mol_runtime.save_checkpoint(checkpoint_path)
    print(f"💾 Saved MoL checkpoint to {checkpoint_path}")
    
    # 5. Push to HuggingFace Hub
    repo_base = "your-username/mol-demo"  # Change this!
    
    print("\n🌐 Pushing to HuggingFace Hub...")
    
    try:
        # Option A: Push lightweight MoL runtime
        print("📤 Pushing MoL runtime (lightweight)...")
        runtime_repo = f"{repo_base}-runtime"
        runtime_url = mol_runtime.push_to_hf(
            repo_id=runtime_repo,
            fusion_type="runtime",
            commit_message="Upload MoL runtime with DialoGPT + DistilGPT2 fusion",
            private=True  # Set to False for public repos
        )
        print(f"✅ Runtime uploaded to: {runtime_url}")
        print(f"   Repository size: ~50-100MB (lightweight)")
        
        # Option B: Push fully fused static model
        print("\n📤 Pushing fully fused model (static)...")
        fused_repo = f"{repo_base}-fused"
        fused_url = mol_runtime.push_to_hf(
            repo_id=fused_repo,
            fusion_type="fused",
            fusion_method="weighted_average",
            commit_message="Upload fully fused model (weighted average)",
            private=True  # Set to False for public repos
        )
        print(f"✅ Fused model uploaded to: {fused_url}")
        print(f"   Repository size: ~2GB (full model)")
        
        # Demonstrate different fusion methods
        print("\n📤 Pushing with different fusion method...")
        best_expert_repo = f"{repo_base}-best"
        best_url = mol_runtime.push_to_hf(
            repo_id=best_expert_repo,
            fusion_type="fused",
            fusion_method="best_expert",
            commit_message="Upload best expert model",
            private=True
        )
        print(f"✅ Best expert model uploaded to: {best_url}")
        
    except Exception as e:
        print(f"❌ Error pushing to HuggingFace: {e}")
        print("💡 Make sure you're logged in to HuggingFace:")
        print("   huggingface-cli login")
        print("   Or set HF_TOKEN environment variable")
        
        # Show how to create fused model locally
        print("\n🔧 Creating fused model locally instead...")
        fused_model = mol_runtime.create_fused_model("weighted_average")
        print(f"✅ Created local fused model with {sum(p.numel() for p in fused_model.parameters()):,} parameters")
    
    # 6. Usage examples
    print("\n💡 Usage Examples:")
    print(f"""
# Loading MoL runtime:
from mol import MoLRuntime
mol_model = MoLRuntime.load_checkpoint("{checkpoint_path}")

# Loading from HuggingFace (runtime):
# mol_model = MoLRuntime.from_pretrained("{runtime_repo}")  # Future feature

# Loading fused model from HuggingFace:
# from transformers import AutoModel, AutoTokenizer
# model = AutoModel.from_pretrained("{fused_repo}")
# tokenizer = AutoTokenizer.from_pretrained("{fused_repo}")
""")
    
    # Cleanup
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"🧹 Cleaned up {checkpoint_path}")


def train_mol_system(mol_runtime):
    """Optional: Train the MoL system (simplified for demo)."""
    # Create simple training data
    train_dataset = create_simple_dataset(
        ["Hello world!", "How are you?", "Nice to meet you!"] * 10,
        mol_runtime.tokenizer,
        max_length=32
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    # Training configuration
    training_config = TrainingConfig(
        learning_rate=1e-4,
        router_learning_rate=1e-5,
        batch_size=2,
        max_epochs=1,
        max_steps=10,  # Very short for demo
        warmup_steps=2,
        logging_steps=5,
        output_dir="./mol_demo_training",
        use_wandb=False
    )
    
    # Train
    trainer = MoLTrainer(mol_runtime, training_config)
    trainer.setup_optimization()
    trainer.freeze_experts()
    trainer.train(train_dataloader)
    
    print("✅ Training completed!")


def cli_examples():
    """Show CLI usage examples."""
    print("\n🖥️  CLI Usage Examples:")
    print("""
# Push MoL runtime (lightweight):
mol-merge push-hf my-username/my-mol-model \\
  --mol-checkpoint ./mol_checkpoint.pt \\
  --fusion-type runtime \\
  --commit-message "Upload MoL runtime"

# Push fully fused model:
mol-merge push-hf my-username/my-fused-model \\
  --mol-checkpoint ./mol_checkpoint.pt \\
  --fusion-type fused \\
  --fusion-method weighted_average \\
  --commit-message "Upload fused model"

# Push with custom token:
mol-merge push-hf my-username/my-model \\
  --mol-checkpoint ./mol_checkpoint.pt \\
  --token hf_xxxxxxxxxx \\
  --private
""")


if __name__ == "__main__":
    # Show CLI examples first
    cli_examples()
    
    print("\n" + "="*60)
    print("🎯 INTERACTIVE DEMO")
    print("="*60)
    print("⚠️  Note: Update 'your-username' in the script before running!")
    print("⚠️  Make sure you're logged in: huggingface-cli login")
    
    try:
        create_and_push_mol_model()
        print("\n🎉 Demo completed successfully!")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()