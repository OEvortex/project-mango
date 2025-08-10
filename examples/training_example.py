"""
Training example for MoL system.
"""

import torch
from torch.utils.data import DataLoader
from mol import MoLRuntime
from mol.core.mol_runtime import MoLConfig
from mol.training.trainer import MoLTrainer, TrainingConfig, create_simple_dataset


def training_example():
    """Demonstrate MoL training pipeline."""
    print("🎯 MoL Training Example")
    print("=" * 50)
    
    # Configuration for MoL system
    mol_config = MoLConfig(
        models=[
            "microsoft/DialoGPT-small",  # ~117M parameters
            "distilgpt2",                # ~82M parameters
        ],
        adapter_type="linear",
        router_type="simple",
        max_layers=2,  # Only use 2 layers for quick training demo
        temperature=1.0,
        entropy_penalty_coeff=0.1,
        load_balancing_coeff=0.01,
        memory_efficient=True
    )
    
    # Training configuration
    training_config = TrainingConfig(
        learning_rate=1e-4,
        router_learning_rate=1e-5,
        weight_decay=0.01,
        batch_size=2,  # Small batch for demo
        max_epochs=1,
        max_steps=50,  # Just a few steps for demo
        warmup_steps=5,
        logging_steps=5,
        eval_steps=20,
        save_steps=25,
        gradient_clip_norm=1.0,
        entropy_penalty_coeff=0.1,
        load_balancing_coeff=0.01,
        freeze_experts=True,
        output_dir="./mol_training_demo",
        run_name="mol_demo_training",
        use_wandb=False  # Disable for demo
    )
    
    print(f"MoL config: {mol_config.models}")
    print(f"Training config: max_steps={training_config.max_steps}")
    
    # Create MoL runtime
    print("\n🚀 Initializing MoL Runtime...")
    mol_runtime = MoLRuntime(mol_config)
    
    # Setup embeddings and LM head
    print("📝 Setting up embeddings and LM head...")
    mol_runtime.setup_embeddings()
    mol_runtime.setup_lm_head()
    
    # Add fusion layers
    print("\n🔧 Adding MoL fusion layers...")
    
    # Layer 0: Mix early layers from both models
    mol_runtime.add_layer([
        ("microsoft/DialoGPT-small", 0),
        ("distilgpt2", 0)
    ], layer_idx=0)
    
    # Layer 1: Mix different layers for diversity
    mol_runtime.add_layer([
        ("microsoft/DialoGPT-small", 2),
        ("distilgpt2", 1)
    ], layer_idx=1)
    
    print(f"Added {len(mol_runtime.layers)} MoL layers")
    
    # Create simple training data
    print("\n📚 Creating training dataset...")
    train_texts = [
        "Hello, how are you today?",
        "What is your favorite color?",
        "Tell me about artificial intelligence.",
        "The weather is nice today.",
        "I love reading books about science.",
        "Technology is advancing rapidly.",
        "Machine learning is fascinating.",
        "Natural language processing is complex.",
        "Deep learning models are powerful.",
        "Transformers changed NLP forever."
    ]
    
    eval_texts = [
        "How do you feel about this?",
        "What do you think about that?",
        "Science is interesting to study.",
        "AI will change the world."
    ]
    
    # Create datasets
    train_dataset = create_simple_dataset(train_texts, mol_runtime.tokenizer, max_length=64)
    eval_dataset = create_simple_dataset(eval_texts, mol_runtime.tokenizer, max_length=64)
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=training_config.batch_size, 
        shuffle=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=training_config.batch_size, 
        shuffle=False
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    
    # Initialize trainer
    print("\n🏋️ Initializing trainer...")
    trainer = MoLTrainer(mol_runtime, training_config)
    
    # Print model statistics
    from mol.utils.model_utils import ModelUtils
    total_params = ModelUtils.count_parameters(mol_runtime)
    trainable_params = ModelUtils.count_parameters(mol_runtime, trainable_only=True)
    
    print(f"\nModel Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")
    
    # Start training
    print("\n🚀 Starting training...")
    try:
        trainer.train(train_dataloader, eval_dataloader)
        print("\n✅ Training completed successfully!")
        
        # Test generation after training
        print("\n🎭 Testing generation after training...")
        test_text = "Hello, how are you"
        inputs = mol_runtime.tokenizer(
            test_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        
        mol_runtime.eval()
        with torch.no_grad():
            try:
                generated = mol_runtime.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=10,
                    temperature=0.8,
                    do_sample=True
                )
                
                generated_text = mol_runtime.tokenizer.decode(generated[0], skip_special_tokens=True)
                print(f"Input: '{test_text}'")
                print(f"Generated: '{generated_text}'")
                
            except Exception as e:
                print(f"Generation failed (expected for demo): {e}")
        
        return trainer
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the training example
    trainer = training_example()