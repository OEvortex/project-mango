"""
Basic demo of MoL layer fusion.
"""

import torch
from mol import MoLRuntime
from mol.core.mol_runtime import MoLConfig


def basic_fusion_demo():
    """Demonstrate basic layer fusion between two models."""
    print("🧠 MoL Basic Fusion Demo")
    print("=" * 50)
    
    # Configuration for MoL system
    config = MoLConfig(
        models=[
            "microsoft/DialoGPT-small",  # ~117M parameters
            "distilgpt2",                # ~82M parameters
        ],
        adapter_type="linear",
        router_type="simple",
        max_layers=4,  # Only use first 4 layers for demo
        temperature=1.0,
        memory_efficient=True
    )
    
    print(f"Models to fuse: {config.models}")
    print(f"Adapter type: {config.adapter_type}")
    print(f"Router type: {config.router_type}")
    
    # Create MoL runtime
    print("\n🚀 Initializing MoL Runtime...")
    mol_runtime = MoLRuntime(config)
    
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
    
    # Layer 1: Mix middle layers
    mol_runtime.add_layer([
        ("microsoft/DialoGPT-small", 2),
        ("distilgpt2", 1)
    ], layer_idx=1)
    
    # Layer 2: Use different layers for specialization
    mol_runtime.add_layer([
        ("microsoft/DialoGPT-small", 4),
        ("distilgpt2", 2)
    ], layer_idx=2)
    
    print(f"Added {len(mol_runtime.layers)} MoL layers")
    
    # Demonstrate inference
    print("\n💭 Running inference demo...")
    
    # Prepare sample input
    text = "Hello, how are you today?"
    inputs = mol_runtime.tokenizer(
        text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    )
    
    print(f"Input text: '{text}'")
    print(f"Input tokens: {inputs['input_ids'].shape}")
    
    # Forward pass with routing statistics
    with torch.no_grad():
        mol_runtime.eval()
        hidden_states, router_stats = mol_runtime.forward(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            return_router_stats=True
        )
    
    print(f"Output shape: {hidden_states.shape}")
    
    # Display router statistics
    if router_stats:
        print("\n📊 Router Statistics:")
        for layer_name, stats in router_stats.items():
            print(f"  {layer_name}:")
            print(f"    Router entropy: {stats['router_entropy']:.4f}")
            print(f"    Load balancing loss: {stats['load_balancing_loss']:.4f}")
            print(f"    Expert usage: {stats['expert_weights'].mean(dim=(0,1)).tolist()}")
    
    # Generate text
    print("\n✨ Generating text...")
    try:
        generated = mol_runtime.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=20,
            temperature=0.8,
            do_sample=True
        )
        
        generated_text = mol_runtime.tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"Generated: '{generated_text}'")
        
    except Exception as e:
        print(f"Text generation failed: {e}")
        print("This is expected for this basic demo - full generation requires more setup")
    
    # Model statistics
    print("\n📈 Model Statistics:")
    from mol.utils.model_utils import ModelUtils
    
    total_params = ModelUtils.count_parameters(mol_runtime)
    trainable_params = ModelUtils.count_parameters(mol_runtime, trainable_only=True)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Adapter/Router parameters: {trainable_params:,}")
    
    # Memory usage
    print("\n💾 Memory Usage:")
    memory_stats = mol_runtime.memory_manager.get_memory_stats()
    print(f"RAM usage: {memory_stats.used_ram:.1f}GB / {memory_stats.total_ram:.1f}GB")
    if memory_stats.total_vram > 0:
        print(f"VRAM usage: {memory_stats.used_vram:.1f}GB / {memory_stats.total_vram:.1f}GB")
    
    print("\n✅ Demo completed successfully!")
    return mol_runtime


if __name__ == "__main__":
    # Run the demo
    runtime = basic_fusion_demo()