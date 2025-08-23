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
    
    # Layer 0: Mix early layers from both models
    mol_runtime.add_layer([
        ("suayptalha/Qwen3-0.6B-Medical-Expert", 0),
        ("Qwen/Qwen3-0.6B", 0)
    ], layer_idx=0)
    
    # Layer 1: Mix different layers for diversity
    mol_runtime.add_layer([
        ("suayptalha/Qwen3-0.6B-Medical-Expert", 2),
        ("Qwen/Qwen3-0.6B", 1)
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
        "Artificial neural networks learn from data.",
        "Machine learning algorithms improve with experience.",
        "Deep learning uses multi-layered neural networks.",
        "Natural language understanding processes human speech.",
        "Computer graphics create realistic visual effects.",
        "Distributed systems handle large-scale computing.",
        "Information security prevents unauthorized access.",
        "Software testing ensures application quality.",
        "Version control systems track code changes.",
        "Agile methodology emphasizes iterative development.",
        "DevOps practices integrate development and operations.",
        "Cloud services provide on-demand computing resources.",
        "Microservices architecture promotes modular design.",
        "API development enables system integration.",
        "Data visualization makes information accessible.",
        "Statistical analysis reveals data insights.",
        "Predictive modeling forecasts future trends.",
        "Time series analysis examines temporal patterns.",
        "Regression analysis studies variable relationships.",
        "Classification algorithms categorize data points.",
        "Clustering techniques group similar items.",
        "Recommendation systems suggest relevant content.",
        "Search engines index and retrieve information.",
        "Operating systems manage computer resources.",
        "Network protocols enable communication standards.",
        "Distributed computing spreads workload across systems.",
        "Parallel processing executes tasks simultaneously.",
        "Concurrent programming handles multiple threads.",
        "Functional programming emphasizes pure functions.",
        "Object-oriented programming organizes code into classes.",
        "Procedural programming follows step-by-step instructions.",
        "Dynamic programming optimizes recursive solutions.",
        "Algorithm design creates efficient problem-solving methods.",
        "Data structures organize information systematically.",
        "Computational complexity measures algorithm efficiency.",
        "Graph theory studies network relationships.",
        "Linear algebra provides mathematical foundations.",
        "Calculus enables optimization and analysis.",
        "Statistics supports data-driven decision making.",
        "Probability theory models uncertain events.",
        "Digital signal processing manipulates electronic signals.",
        "Image processing enhances and analyzes pictures.",
        "Audio processing handles sound and music.",
        "Video processing manages moving images.",
        "Computer animation creates lifelike movements.",
        "3D modeling builds virtual three-dimensional objects.",
        "Simulation software models real-world phenomena.",
        "Mathematical modeling describes complex systems.",
        "Optimization algorithms find best solutions.",
        "Heuristic methods provide approximate solutions.",
        "Genetic algorithms evolve solutions over generations.",
        "Swarm intelligence mimics collective behavior.",
        "Reinforcement learning learns through trial and error.",
        "Supervised learning uses labeled training data.",
        "Unsupervised learning discovers hidden patterns.",
        "Semi-supervised learning combines labeled and unlabeled data.",
        "Transfer learning applies knowledge across domains.",
        "Few-shot learning works with limited examples.",
        "Zero-shot learning generalizes without specific training.",
        "Multi-task learning handles multiple objectives simultaneously.",
        "Ensemble methods combine multiple model predictions.",
        "Cross-validation evaluates model performance reliably.",
        "Hyperparameter tuning optimizes model configuration.",
        "Feature engineering creates informative input variables.",
        "Data preprocessing cleans and prepares datasets.",
        "Exploratory data analysis reveals initial insights.",
        "Statistical inference draws conclusions from samples.",
        "Hypothesis testing validates scientific claims.",
        "Experimental design controls for confounding variables."
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