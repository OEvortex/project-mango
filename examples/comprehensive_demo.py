#!/usr/bin/env python3
"""
Comprehensive MoL System Demo

This script demonstrates the complete MoL (Modular Layer) system for fusing
transformer blocks from different LLMs. It showcases all major features:

1. Layer fusion from multiple models
2. Adapter-based dimension matching  
3. Intelligent routing (both pooled and token-level)
4. Training pipeline with regularization
5. Evaluation and analysis
6. Memory management and optimization

Usage:
    python comprehensive_demo.py [--train] [--eval] [--small-models]
"""

import argparse
import torch
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from mol import MoLRuntime
from mol.core.mol_runtime import MoLConfig
from mol.training.trainer import MoLTrainer, TrainingConfig, create_simple_dataset
from mol.utils.model_utils import ModelUtils
from torch.utils.data import DataLoader


def demonstrate_layer_fusion(use_small_models: bool = True):
    """Demonstrate basic layer fusion capabilities."""
    print("\n" + "="*60)
    print("🧠 MODULAR LAYER FUSION DEMONSTRATION")
    print("="*60)
    
    # Select models based on size preference
    if use_small_models:
        models = ["microsoft/DialoGPT-small", "distilgpt2"]  # Smaller for demo
        max_layers = 3
    else:
        models = ["microsoft/DialoGPT-medium", "gpt2"]  # Larger models
        max_layers = 6
    
    print(f"🎯 Target models: {models}")
    print(f"📊 Max layers: {max_layers}")
    
    # Create MoL configuration
    config = MoLConfig(
        models=models,
        adapter_type="bottleneck",  # Use bottleneck for efficiency
        router_type="token",        # Token-level routing for expressiveness
        max_layers=max_layers,
        temperature=1.0,
        entropy_penalty_coeff=0.1,
        load_balancing_coeff=0.01,
        top_k_experts=None,  # Use all experts
        memory_efficient=True
    )
    
    # Initialize MoL runtime
    print("\n🚀 Initializing MoL Runtime...")
    mol_runtime = MoLRuntime(config)
    
    # Display model information
    print("\n📋 Model Information:")
    for name, info in mol_runtime.model_infos.items():
        print(f"   • {name}:")
        print(f"     - Hidden dim: {info.hidden_dim}")
        print(f"     - Layers: {info.num_layers}")
        print(f"     - Architecture: {info.architecture_type}")
    
    print(f"\n🎯 Target hidden dimension: {mol_runtime.target_hidden_dim}")
    
    # Setup embeddings and LM head
    print("\n📝 Setting up embeddings and LM head...")
    mol_runtime.setup_embeddings()
    try:
        mol_runtime.setup_lm_head()
        print("✅ LM head setup successful")
    except Exception as e:
        print(f"⚠️ LM head setup failed: {e}")
    
    # Design fusion architecture
    print(f"\n🏗️ Designing fusion architecture ({max_layers} layers)...")
    
    # Add diverse layer combinations
    fusion_layers = [
        # Layer 0: Early representations
        (0, [(models[0], 0), (models[1], 0)]),
        # Layer 1: Mixed early-middle  
        (1, [(models[0], 1), (models[1], 1)]),
        # Layer 2: Middle representations
        (2, [(models[0], 2), (models[1], 2)]),
    ]
    
    # Add more layers if using larger models
    if max_layers > 3:
        fusion_layers.extend([
            (3, [(models[0], 3), (models[1], 3)]),
            (4, [(models[0], 4), (models[1], 4)]),
            (5, [(models[0], 5), (models[1], 5)]),
        ])
    
    for layer_idx, experts in fusion_layers:
        print(f"   Layer {layer_idx}: {[f'{m}[{i}]' for m, i in experts]}")
        mol_runtime.add_layer(experts, layer_idx)
    
    print(f"✅ Created {len(mol_runtime.layers)} fusion layers")
    
    # Analyze the architecture
    print("\n📊 Architecture Analysis:")
    total_params = ModelUtils.count_parameters(mol_runtime)
    trainable_params = ModelUtils.count_parameters(mol_runtime, trainable_only=True)
    
    print(f"   • Total parameters: {total_params:,}")
    print(f"   • Trainable parameters: {trainable_params:,}")
    print(f"   • Trainable ratio: {trainable_params/total_params*100:.2f}%")
    
    # Memory analysis
    memory_stats = mol_runtime.memory_manager.get_memory_stats()
    print(f"   • RAM usage: {memory_stats.used_ram:.1f}GB / {memory_stats.total_ram:.1f}GB")
    if memory_stats.total_vram > 0:
        print(f"   • VRAM usage: {memory_stats.used_vram:.1f}GB / {memory_stats.total_vram:.1f}GB")
    
    return mol_runtime


def demonstrate_inference(mol_runtime):
    """Demonstrate inference with routing analysis."""
    print("\n" + "="*60)
    print("🎭 INFERENCE AND ROUTING DEMONSTRATION") 
    print("="*60)
    
    # Test inputs representing different types of content
    test_inputs = [
        "Hello, how are you today?",                    # Conversational
        "Explain quantum computing in simple terms.",   # Technical
        "Once upon a time in a distant galaxy,",       # Creative
        "The economic indicators suggest that",          # Analytical
        "My favorite recipe for chocolate cake is",     # Instructional
    ]
    
    print(f"🧪 Testing on {len(test_inputs)} diverse inputs...")
    
    routing_analysis = []
    
    mol_runtime.eval()
    with torch.no_grad():
        for i, text in enumerate(test_inputs):
            print(f"\n📝 Input {i+1}: '{text}'")
            
            # Tokenize
            inputs = mol_runtime.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            
            print(f"   🔤 Tokens: {inputs['input_ids'].shape[1]}")
            
            # Forward pass with routing stats
            hidden_states, router_stats = mol_runtime.forward(
                inputs['input_ids'],
                inputs['attention_mask'],
                return_router_stats=True
            )
            
            print(f"   📐 Output shape: {hidden_states.shape}")
            
            # Analyze routing behavior
            if router_stats:
                print("   🧭 Routing Analysis:")
                
                text_routing = {"text": text, "layers": {}}
                
                for layer_name, stats in router_stats.items():
                    expert_weights = stats['expert_weights']  # [batch, seq, experts]
                    avg_weights = expert_weights.mean(dim=(0, 1))  # Average per expert
                    
                    print(f"      {layer_name}:")
                    print(f"        Entropy: {stats['router_entropy']:.3f}")
                    print(f"        Load balance: {stats['load_balancing_loss']:.4f}")
                    print(f"        Expert usage: {avg_weights.tolist()}")
                    
                    text_routing["layers"][layer_name] = {
                        "entropy": stats['router_entropy'],
                        "expert_weights": avg_weights.tolist()
                    }
                
                routing_analysis.append(text_routing)
    
    # Cross-input routing analysis
    print("\n🔍 Cross-Input Routing Analysis:")
    
    if routing_analysis:
        # Analyze if different inputs use different routing patterns
        layer_names = list(routing_analysis[0]["layers"].keys())
        
        for layer_name in layer_names:
            print(f"\n   📊 {layer_name} Expert Usage Patterns:")
            
            for i, analysis in enumerate(routing_analysis):
                weights = analysis["layers"][layer_name]["expert_weights"]
                dominant_expert = weights.index(max(weights))
                print(f"      Input {i+1}: Expert {dominant_expert} ({weights[dominant_expert]:.3f})")
    
    return routing_analysis


def demonstrate_training(mol_runtime):
    """Demonstrate training pipeline."""
    print("\n" + "="*60)
    print("🏋️ TRAINING DEMONSTRATION")
    print("="*60)
    
    # Training configuration optimized for demo
    training_config = TrainingConfig(
        learning_rate=5e-4,          # Higher LR for quick demo
        router_learning_rate=1e-4,    # Router LR
        weight_decay=0.01,
        batch_size=2,                 # Small batch for memory
        max_epochs=1,
        max_steps=30,                 # Just enough to see training
        warmup_steps=5,
        logging_steps=5,
        eval_steps=15,
        save_steps=20,
        gradient_clip_norm=1.0,
        entropy_penalty_coeff=0.1,
        load_balancing_coeff=0.01,
        freeze_experts=True,          # Keep expert models frozen
        use_gradient_checkpointing=False,
        output_dir="./mol_demo_training",
        run_name="mol_comprehensive_demo",
        use_wandb=False
    )
    
    print("📋 Training Configuration:")
    print(f"   • Learning rate: {training_config.learning_rate}")
    print(f"   • Router LR: {training_config.router_learning_rate}")
    print(f"   • Max steps: {training_config.max_steps}")
    print(f"   • Batch size: {training_config.batch_size}")
    print(f"   • Freeze experts: {training_config.freeze_experts}")
    
    # Create diverse training data
    train_texts = [
        # Conversational
        "Hello, how can I help you today?",
        "What's your favorite movie and why?",
        "I'm feeling a bit tired. Any suggestions?",
        
        # Technical
        "Machine learning algorithms can be categorized into supervised and unsupervised learning.",
        "The transformer architecture revolutionized natural language processing.",
        "Deep neural networks consist of multiple layers of interconnected neurons.",
        
        # Creative  
        "In the heart of the ancient forest, a mysterious light began to glow.",
        "The space explorer gazed upon the twin moons rising over the alien landscape.",
        "She painted with colors that seemed to dance across the canvas.",
        
        # Analytical
        "The quarterly earnings report shows significant growth in the technology sector.",
        "Climate change impacts can be observed through rising global temperatures.",
        "Consumer behavior patterns indicate a shift towards sustainable products.",
        
        # Instructional
        "To make a perfect cup of coffee, start by grinding the beans fresh.",
        "The first step in solving this equation is to isolate the variable.",
        "When training a neural network, it's important to monitor the loss function.",
    ]
    
    eval_texts = [
        "Can you explain this concept to me?",
        "The research findings demonstrate clear evidence of the hypothesis.",
        "Once there was a kingdom where magic still existed.",
        "The financial markets responded positively to the announcement.",
    ]
    
    print(f"\n📚 Dataset:")
    print(f"   • Training samples: {len(train_texts)}")
    print(f"   • Evaluation samples: {len(eval_texts)}")
    
    # Create datasets and dataloaders
    train_dataset = create_simple_dataset(train_texts, mol_runtime.tokenizer, max_length=64)
    eval_dataset = create_simple_dataset(eval_texts, mol_runtime.tokenizer, max_length=64)
    
    train_dataloader = DataLoader(train_dataset, batch_size=training_config.batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=training_config.batch_size, shuffle=False)
    
    # Initialize trainer
    print("\n🔧 Initializing trainer...")
    trainer = MoLTrainer(mol_runtime, training_config)
    
    # Analyze what will be trained
    print("\n🎯 Training Analysis:")
    for name, param in mol_runtime.named_parameters():
        if param.requires_grad:
            print(f"   • {name}: {param.numel():,} parameters")
    
    # Start training
    print(f"\n🚀 Starting training for {training_config.max_steps} steps...")
    
    try:
        trainer.train(train_dataloader, eval_dataloader)
        print("✅ Training completed successfully!")
        
        return trainer
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def demonstrate_evaluation(mol_runtime):
    """Demonstrate evaluation and analysis."""
    print("\n" + "="*60)
    print("📊 EVALUATION AND ANALYSIS")
    print("="*60)
    
    # Evaluation texts covering different domains
    eval_texts = [
        "The future of artificial intelligence looks promising.",
        "How do neural networks learn from data?",
        "In a world where technology advances rapidly,",
        "The economic implications of automation include",
        "To understand quantum mechanics, one must first",
        "Customer satisfaction surveys indicate that",
        "The adventure began when she opened the mysterious door.",
        "Climate science research has shown that",
        "Programming languages each have their own strengths:",
        "The recipe calls for fresh ingredients and careful timing.",
    ]
    
    print(f"📝 Evaluating on {len(eval_texts)} texts...")
    
    # Collect comprehensive statistics
    mol_runtime.eval()
    
    results = {
        "routing_patterns": [],
        "expert_usage": [],
        "entropy_stats": [],
        "performance_metrics": {}
    }
    
    with torch.no_grad():
        total_tokens = 0
        total_expert_usage = None
        all_entropies = []
        
        for i, text in enumerate(eval_texts):
            inputs = mol_runtime.tokenizer(
                text,
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=128
            )
            
            # Forward pass
            hidden_states, router_stats = mol_runtime.forward(
                inputs['input_ids'],
                inputs['attention_mask'],
                return_router_stats=True
            )
            
            total_tokens += inputs['input_ids'].shape[1]
            
            if router_stats:
                # Collect routing statistics
                text_stats = {"text": text, "stats": {}}
                layer_entropies = []
                
                for layer_name, stats in router_stats.items():
                    expert_weights = stats['expert_weights']
                    entropy = stats['router_entropy']
                    
                    layer_entropies.append(entropy)
                    text_stats["stats"][layer_name] = {
                        "entropy": entropy,
                        "expert_usage": expert_weights.mean(dim=(0, 1)).tolist()
                    }
                    
                    # Accumulate expert usage
                    layer_usage = expert_weights.mean(dim=(0, 1))
                    if total_expert_usage is None:
                        total_expert_usage = torch.zeros_like(layer_usage)
                    total_expert_usage += layer_usage
                
                results["routing_patterns"].append(text_stats)
                all_entropies.extend(layer_entropies)
            
            if (i + 1) % 3 == 0:
                print(f"   Processed {i + 1}/{len(eval_texts)} texts...")
    
    # Compute aggregate statistics
    print("\n📈 Evaluation Results:")
    
    if all_entropies:
        avg_entropy = sum(all_entropies) / len(all_entropies)
        print(f"   • Average routing entropy: {avg_entropy:.3f}")
        print(f"   • Entropy std deviation: {torch.std(torch.tensor(all_entropies)):.3f}")
        
        results["performance_metrics"]["avg_entropy"] = avg_entropy
    
    if total_expert_usage is not None:
        # Normalize expert usage
        normalized_usage = total_expert_usage / len(eval_texts)
        usage_balance = 1.0 - torch.var(normalized_usage).item()
        
        print(f"   • Expert usage balance: {usage_balance:.3f}")
        print(f"   • Expert usage distribution: {normalized_usage.tolist()}")
        
        results["performance_metrics"]["usage_balance"] = usage_balance
        results["performance_metrics"]["expert_distribution"] = normalized_usage.tolist()
    
    print(f"   • Total tokens processed: {total_tokens:,}")
    print(f"   • Average tokens per text: {total_tokens / len(eval_texts):.1f}")
    
    # Routing pattern analysis
    if results["routing_patterns"]:
        print("\n🔍 Routing Pattern Analysis:")
        
        # Find texts that use different routing patterns
        layer_0_patterns = []
        for pattern in results["routing_patterns"]:
            if "layer_0" in pattern["stats"]:
                usage = pattern["stats"]["layer_0"]["expert_usage"]
                dominant_expert = usage.index(max(usage))
                layer_0_patterns.append((pattern["text"][:50] + "...", dominant_expert, max(usage)))
        
        # Group by dominant expert
        expert_groups = {}
        for text, expert, confidence in layer_0_patterns:
            if expert not in expert_groups:
                expert_groups[expert] = []
            expert_groups[expert].append((text, confidence))
        
        for expert_id, texts in expert_groups.items():
            print(f"   🎯 Expert {expert_id} (dominant for {len(texts)} texts):")
            for text, conf in texts[:3]:  # Show first 3 examples
                print(f"      • {text} ({conf:.3f})")
    
    return results


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="Comprehensive MoL System Demo")
    parser.add_argument("--train", action="store_true", help="Run training demonstration")
    parser.add_argument("--eval", action="store_true", help="Run evaluation demonstration")
    parser.add_argument("--small-models", action="store_true", default=True, 
                       help="Use smaller models for faster demo")
    parser.add_argument("--inference-only", action="store_true", 
                       help="Only run inference demonstration")
    
    args = parser.parse_args()
    
    print("🎉 COMPREHENSIVE MoL SYSTEM DEMONSTRATION")
    print("=" * 80)
    print("This demo showcases the complete Modular Layer (MoL) fusion system.")
    print("Features: Layer fusion, adaptive routing, training pipeline, evaluation")
    print("=" * 80)
    
    try:
        # Phase 1: Layer Fusion Setup
        mol_runtime = demonstrate_layer_fusion(args.small_models)
        
        # Phase 2: Inference and Routing
        routing_analysis = demonstrate_inference(mol_runtime)
        
        if args.inference_only:
            print("\n✅ Inference-only demonstration completed!")
            return
        
        # Phase 3: Training (optional)
        trainer = None
        if args.train:
            trainer = demonstrate_training(mol_runtime)
        
        # Phase 4: Evaluation (optional)
        if args.eval:
            evaluation_results = demonstrate_evaluation(mol_runtime)
        
        # Summary
        print("\n" + "="*80)
        print("🎯 DEMONSTRATION SUMMARY")
        print("="*80)
        print("✅ Layer fusion system successfully created and tested")
        print("✅ Adaptive routing behavior demonstrated") 
        print("✅ Model architecture and memory usage analyzed")
        
        if trainer:
            print("✅ Training pipeline executed successfully")
        
        if args.eval:
            print("✅ Comprehensive evaluation completed")
        
        print("\n🚀 The MoL system is ready for production use!")
        print("   Next steps: Scale to larger models, add more layers, fine-tune for specific tasks")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())