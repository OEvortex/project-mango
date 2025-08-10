"""
Simple evaluation script for MoL system.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np
from mol import MoLRuntime
from mol.core.mol_runtime import MoLConfig


def evaluate_mol_system(
    mol_runtime: MoLRuntime,
    test_texts: List[str],
    compute_perplexity: bool = True,
    analyze_routing: bool = True
) -> Dict[str, float]:
    """
    Evaluate MoL system on test texts.
    
    Args:
        mol_runtime: The MoL runtime to evaluate
        test_texts: List of test texts
        compute_perplexity: Whether to compute perplexity
        analyze_routing: Whether to analyze routing behavior
    
    Returns:
        Dictionary of evaluation metrics
    """
    mol_runtime.eval()
    results = {}
    
    if not mol_runtime.lm_head:
        print("⚠️ LM head not available. Skipping perplexity computation.")
        compute_perplexity = False
    
    # Collect statistics
    perplexities = []
    routing_entropies = []
    expert_usage_stats = []
    
    print(f"📊 Evaluating on {len(test_texts)} texts...")
    
    with torch.no_grad():
        for i, text in enumerate(test_texts):
            # Tokenize input
            inputs = mol_runtime.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            )
            
            # Forward pass with router statistics
            hidden_states, router_stats = mol_runtime.forward(
                inputs['input_ids'],
                inputs['attention_mask'],
                return_router_stats=True
            )
            
            # Compute perplexity if LM head is available
            if compute_perplexity:
                logits = mol_runtime.lm_head(hidden_states)
                
                # Compute cross-entropy loss for perplexity
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs['input_ids'][..., 1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                perplexity = torch.exp(loss).item()
                perplexities.append(perplexity)
            
            # Analyze routing behavior
            if analyze_routing and router_stats:
                layer_entropies = []
                layer_usage = []
                
                for layer_name, stats in router_stats.items():
                    layer_entropies.append(stats['router_entropy'])
                    
                    # Expert usage distribution
                    expert_weights = stats['expert_weights']  # [batch, seq, num_experts]
                    avg_usage = expert_weights.mean(dim=(0, 1))  # Average usage per expert
                    layer_usage.append(avg_usage.cpu().numpy())
                
                routing_entropies.append(np.mean(layer_entropies))
                expert_usage_stats.append(layer_usage)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(test_texts)} texts...")
    
    # Compute aggregate statistics
    if perplexities:
        results['perplexity_mean'] = np.mean(perplexities)
        results['perplexity_std'] = np.std(perplexities)
        results['perplexity_median'] = np.median(perplexities)
        
        print(f"📈 Perplexity: {results['perplexity_mean']:.2f} ± {results['perplexity_std']:.2f}")
    
    if routing_entropies:
        results['routing_entropy_mean'] = np.mean(routing_entropies)
        results['routing_entropy_std'] = np.std(routing_entropies)
        
        print(f"🎯 Router Entropy: {results['routing_entropy_mean']:.3f} ± {results['routing_entropy_std']:.3f}")
    
    if expert_usage_stats:
        # Analyze expert usage distribution
        all_usage = np.array(expert_usage_stats)  # [num_texts, num_layers, num_experts]
        
        # Average expert usage across all texts and layers
        mean_usage_per_expert = all_usage.mean(axis=(0, 1))  # [num_experts]
        usage_variance = np.var(mean_usage_per_expert)
        
        results['expert_usage_variance'] = usage_variance
        results['expert_usage_balance'] = 1.0 - usage_variance  # Higher is more balanced
        
        print(f"⚖️ Expert Usage Balance: {results['expert_usage_balance']:.3f}")
        print(f"   Expert usage distribution: {mean_usage_per_expert}")
    
    return results


def compare_with_baseline(
    mol_runtime: MoLRuntime,
    baseline_texts: List[str],
    mol_texts: List[str]
) -> Dict[str, float]:
    """
    Compare MoL system performance with baseline.
    This is a simplified comparison for demonstration.
    """
    print("🔄 Comparing MoL with baseline...")
    
    # Evaluate baseline (first model only)
    print("Evaluating baseline (first model)...")
    
    # For demo purposes, we'll use a simple metric
    # In practice, you'd want to compare against individual models
    
    # Simplified comparison - just return some dummy metrics for demo
    comparison_results = {
        'mol_vs_baseline_perplexity_ratio': 0.95,  # Lower is better
        'mol_routing_efficiency': 0.85,  # Higher is better
        'mol_expert_utilization': 0.78,  # Higher means more balanced usage
    }
    
    print("📊 Comparison Results:")
    for metric, value in comparison_results.items():
        print(f"   {metric}: {value:.3f}")
    
    return comparison_results


def evaluation_demo():
    """Demonstrate MoL evaluation."""
    print("📊 MoL Evaluation Demo")
    print("=" * 50)
    
    # Create a simple MoL system for evaluation
    config = MoLConfig(
        models=[
            "microsoft/DialoGPT-small",
            "distilgpt2",
        ],
        adapter_type="linear",
        router_type="simple",
        max_layers=2,
        temperature=1.0
    )
    
    print("🚀 Initializing MoL Runtime...")
    mol_runtime = MoLRuntime(config)
    
    # Setup components
    mol_runtime.setup_embeddings()
    try:
        mol_runtime.setup_lm_head()
    except Exception as e:
        print(f"⚠️ Could not setup LM head: {e}")
    
    # Add some fusion layers
    mol_runtime.add_layer([
        ("microsoft/DialoGPT-small", 0),
        ("distilgpt2", 0)
    ], layer_idx=0)
    
    mol_runtime.add_layer([
        ("microsoft/DialoGPT-small", 1),
        ("distilgpt2", 1)
    ], layer_idx=1)
    
    # Test texts for evaluation
    test_texts = [
        "Hello, how are you today?",
        "What is artificial intelligence?",
        "The weather is beautiful.",
        "Machine learning is fascinating.",
        "I enjoy reading books.",
        "Technology advances rapidly.",
        "Natural language processing is complex.",
        "Deep learning models are powerful.",
        "Science and research are important.",
        "Communication is key to success."
    ]
    
    print(f"\n📚 Evaluating on {len(test_texts)} test texts...")
    
    # Run evaluation
    try:
        results = evaluate_mol_system(
            mol_runtime,
            test_texts,
            compute_perplexity=bool(mol_runtime.lm_head),
            analyze_routing=True
        )
        
        print("\n✅ Evaluation Results:")
        for metric, value in results.items():
            if isinstance(value, float):
                print(f"   {metric}: {value:.4f}")
            else:
                print(f"   {metric}: {value}")
        
        return results
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


if __name__ == "__main__":
    # Run the evaluation demo
    results = evaluation_demo()