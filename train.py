"""
Training example for MoL system.
"""

import torch
from torch.utils.data import DataLoader
from mol.core.mol_runtime import MoLRuntime
from mol.core.mol_runtime import MoLConfig
from mol.training.trainer import MoLTrainer, TrainingConfig, create_simple_dataset


def training_example():
    """Demonstrate MoL training pipeline."""
    print("🎯 MoL Training Example")
    print("=" * 50)
    
    # Configuration for MoL system
    mol_config = MoLConfig(
        models=['Qwen/Qwen3-0.6B', 'suayptalha/Qwen3-0.6B-Medical-Expert'],
        adapter_type="linear",
        router_type="simple",
        max_layers=7,  # Only use 7 layers for quick training demo
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
        # max_steps=50,  # Just a few steps for demo
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

    # Add 7 layers, mixing layers from both models for diversity
    mol_runtime.add_layer([
        ("suayptalha/Qwen3-0.6B-Medical-Expert", 0),
        ("Qwen/Qwen3-0.6B", 0)
    ], layer_idx=0)
    mol_runtime.add_layer([
        ("suayptalha/Qwen3-0.6B-Medical-Expert", 1),
        ("Qwen/Qwen3-0.6B", 1)
    ], layer_idx=1)
    mol_runtime.add_layer([
        ("suayptalha/Qwen3-0.6B-Medical-Expert", 2),
        ("Qwen/Qwen3-0.6B", 2)
    ], layer_idx=2)
    mol_runtime.add_layer([
        ("suayptalha/Qwen3-0.6B-Medical-Expert", 3),
        ("Qwen/Qwen3-0.6B", 3)
    ], layer_idx=3)
    mol_runtime.add_layer([
        ("suayptalha/Qwen3-0.6B-Medical-Expert", 4),
        ("Qwen/Qwen3-0.6B", 4)
    ], layer_idx=4)
    mol_runtime.add_layer([
        ("suayptalha/Qwen3-0.6B-Medical-Expert", 5),
        ("Qwen/Qwen3-0.6B", 5)
    ], layer_idx=5)
    mol_runtime.add_layer([
        ("suayptalha/Qwen3-0.6B-Medical-Expert", 6),
        ("Qwen/Qwen3-0.6B", 6)
    ], layer_idx=6)

    print(f"Added {len(mol_runtime.layers)} MoL layers")
    
    # Create simple training data
    print("\n📚 Creating training dataset...")
    train_texts = [
        # Medical-related questions and statements
        "What are the symptoms of diabetes?",
        "How is hypertension diagnosed?",
        "What causes chest pain in patients?",
        "Explain the treatment for pneumonia.",
        "What are the side effects of antibiotics?",
        "How does the cardiovascular system work?",
        "What is the difference between type 1 and type 2 diabetes?",
        "How should wounds be properly cleaned?",
        "What are the signs of a heart attack?",
        "Explain how vaccines work in the body.",
        "What causes high blood pressure?",
        "How is cancer typically treated?",
        "What are the symptoms of COVID-19?",
        "How does the immune system fight infections?",
        "What is the purpose of physical therapy?",
        "Explain the importance of regular checkups.",
        "How do pain medications work?",
        "What are the risks of smoking?",
        "How is blood pressure measured?",
        "What causes allergic reactions?",
        "How do antidepressants affect the brain?",
        "What is the role of nutrition in health?",
        "How are broken bones treated?",
        "What causes kidney stones?",
        "How does anesthesia work during surgery?",
        "What are the symptoms of stroke?",
        "How is diabetes managed daily?",
        "What causes migraine headaches?",
        "How do antibiotics fight bacterial infections?",
        "What is the importance of sleep for health?",
        "How are mental health disorders diagnosed?",
        "What causes heart disease?",
        "How do muscles grow and repair?",
        "What are the benefits of exercise?",
        "How does the digestive system process food?",
        "What causes asthma attacks?",
        "How are chronic diseases managed?",
        "What is the role of genetics in disease?",
        "How do hormones affect the body?",
        "What causes inflammation in tissues?",
        # General questions and statements
        "Hello, how are you today?",
        "What is your favorite color?",
        "Tell me about artificial intelligence.",
        "The weather is nice today.",
        "I love reading books about science.",
        "Technology is advancing rapidly.",
        "Machine learning is fascinating.",
        "Natural language processing is complex.",
        "Deep learning models are powerful.",
        "Transformers changed NLP forever.",
        "Python is a versatile programming language.",
        "Data science involves analyzing large datasets.",
        "Cloud computing provides scalable resources.",
        "Cybersecurity protects digital information.",
        "Mobile applications are everywhere today.",
        "Virtual reality creates immersive experiences.",
        "Blockchain technology enables decentralization.",
        "Internet of Things connects everyday objects.",
        "Quantum computing promises exponential speedup.",
        "Renewable energy sources are becoming popular.",
        "Social media platforms connect people globally.",
        "E-commerce has revolutionized shopping habits.",
        "Remote work is becoming the new normal.",
        "Automation improves efficiency and productivity.",
        "Big data analytics reveals hidden patterns.",
        "Neural networks mimic human brain functions.",
        "Computer vision enables machines to see.",
        "Robotics combines hardware and software systems.",
        "Software engineering builds reliable applications.",
        "Database management systems store information efficiently.",
        "Web development creates interactive online experiences.",
        "Mobile development targets smartphone platforms.",
        "Game development combines creativity and technology.",
        "User experience design focuses on usability.",
        "Digital marketing reaches customers online.",
        "Cryptocurrency offers alternative payment methods.",
        "What is the meaning of life?",
        "How do airplanes fly through the sky?",
        "Why do seasons change throughout the year?",
        "What makes music sound pleasant to humans?",
        "How do computers process information so quickly?",
        "What causes earthquakes and natural disasters?",
        "Why do people have different personality types?",
        "How does the internet connect the world?",
        "What makes some foods taste better than others?",
        "Why do we dream during sleep?",
        "How do plants convert sunlight into energy?",
        "What causes ocean tides to rise and fall?",
        "Why do different cultures have unique traditions?",
        "How do satellites orbit around Earth?",
        "What makes some materials conduct electricity better?",
        "Why do people learn languages at different rates?",
        "How do movies create special effects?",
        "What causes different weather patterns globally?",
        "Why do some animals migrate long distances?",
        "How do telescopes help us see distant stars?"
    ]
    
    eval_texts = [
        # Mix of medical and general evaluation texts
        "What are the symptoms of fever?",
        "How do you feel about this?",
        "What causes headaches in patients?",
        "What do you think about that?",
        "How is blood pressure controlled?",
        "Science is interesting to study.",
        "What are the benefits of vaccination?",
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
        test_text = "Yooo, how are u?"
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
        
        # Save the trained model
        print("\n💾 Saving trained model...")
        try:
            # Save using MoLRuntime's built-in checkpoint method
            save_path = "./mol_trained_model"
            mol_runtime.save_checkpoint(save_path, use_safetensors=True)
            print(f"✅ Model saved successfully to {save_path}.safetensors")
            
            # Also save the final trainer checkpoint for full training state
            trainer.save_checkpoint(is_final=True)
            print(f"✅ Final training checkpoint saved to {training_config.output_dir}")
            
        except Exception as save_error:
            print(f"⚠️ Save failed: {save_error}")
            import traceback
            traceback.print_exc()
        
        return trainer
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the training example
    trainer = training_example()