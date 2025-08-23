# Your First Model Fusion ⭐

Learn the fundamentals of MoL fusion by combining two small models step-by-step.

## 🎯 What You'll Learn

By the end of this tutorial, you'll be able to:
- ✅ Set up a basic MoL fusion configuration
- ✅ Combine layers from two different models
- ✅ Generate text using the fused model
- ✅ Understand routing and adapter concepts
- ✅ Analyze fusion performance

## 📋 Prerequisites

- Python 3.8+ with Project Mango installed
- Basic understanding of language models
- 4GB+ RAM (for small models)

## 🚀 Step 1: Choose Your Models

For this tutorial, we'll use two small, compatible models:

- **`microsoft/DialoGPT-small`** (117M parameters) - Conversational model
- **`distilgpt2`** (82M parameters) - Distilled GPT-2 model

These models are ideal for learning because they:
- Load quickly with minimal memory
- Have similar architectures (both GPT-based)
- Demonstrate clear fusion benefits

## 🔧 Step 2: Basic Setup

Create a new Python file `my_first_fusion.py`:

```python
# Import required components
from mol import MoLRuntime, MoLConfig
import torch

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
```

## ⚙️ Step 3: Create Configuration

```python
# Configure the fusion
config = MoLConfig(
    models=[
        "microsoft/DialoGPT-small",  # Primary model (conversational)
        "distilgpt2",                # Secondary model (general text)
    ],
    adapter_type="linear",           # Simple linear transformation
    router_type="simple",           # Pooled routing strategy
    max_layers=2,                   # Use only 2 layers for quick demo
    memory_efficient=True,          # Enable memory optimizations
    temperature=1.0                 # Balanced routing decisions
)

print("✅ Configuration created")
print(f"Models: {config.models}")
print(f"Adapter: {config.adapter_type}")
print(f"Router: {config.router_type}")
```

### 🔍 Configuration Explained

- **`models`**: List of models to fuse (order matters - first model provides embeddings/LM head)
- **`adapter_type="linear"`**: Simple linear projection for dimension matching
- **`router_type="simple"`**: Makes one routing decision per sequence (vs per token)
- **`max_layers=2`**: Limits complexity for faster demonstration
- **`memory_efficient=True`**: Enables lazy loading and memory optimizations

## 🏗️ Step 4: Initialize MoL Runtime

```python
# Initialize the MoL runtime
print("🔄 Initializing MoL runtime...")
mol = MoLRuntime(config)

# The runtime automatically:
# 1. Loads model information (architecture, dimensions, etc.)
# 2. Determines target hidden dimension (max of all models)
# 3. Sets up tokenizer (from first model)

print(f"✅ Runtime initialized")
print(f"Target hidden dim: {mol.target_hidden_dim}")
print(f"Tokenizer vocab size: {mol.tokenizer.vocab_size}")
```

## 🔗 Step 5: Setup Model Components

```python
# Setup embeddings (from first model)
print("⚙️ Setting up embeddings...")
mol.setup_embeddings()

# Setup language model head (from first model)
print("⚙️ Setting up LM head...")
mol.setup_lm_head()

print("✅ Model components ready")
```

### 🔍 Component Setup Explained

- **Embeddings**: Convert tokens to vectors (from DialoGPT-small)
- **LM Head**: Convert hidden states back to token probabilities (from DialoGPT-small)
- **Fusion Layers**: Will be added next (combination of both models)

## 🧩 Step 6: Add Fusion Layers

```python
# Add first fusion layer
print("🔗 Adding fusion layer 0...")
mol.add_layer([
    ("microsoft/DialoGPT-small", 0),  # Layer 0 from DialoGPT
    ("distilgpt2", 0)                 # Layer 0 from DistilGPT-2
], layer_idx=0)

# Add second fusion layer
print("🔗 Adding fusion layer 1...")
mol.add_layer([
    ("microsoft/DialoGPT-small", 1),  # Layer 1 from DialoGPT
    ("distilgpt2", 1)                 # Layer 1 from DistilGPT-2
], layer_idx=1)

print("✅ Fusion layers added")
print(f"Total MoL layers: {len(mol.layers)}")
```

### 🔍 Layer Addition Explained

Each fusion layer contains:
- **2 Expert Blocks**: Original transformer layers from each model
- **2 Adapters**: Linear projections to match dimensions
- **1 Router**: Decides which expert(s) to use for each input

## 🧪 Step 7: Test Basic Inference

```python
# Prepare test input
test_text = "Hello, how are you today?"
print(f"🧪 Testing with: '{test_text}'")

# Tokenize input
inputs = mol.tokenizer(test_text, return_tensors="pt")
print(f"Input IDs shape: {inputs['input_ids'].shape}")

# Forward pass with routing statistics
print("🔄 Running forward pass...")
hidden_states, router_stats = mol.forward(
    inputs['input_ids'], 
    inputs['attention_mask'],
    return_router_stats=True
)

print(f"✅ Forward pass successful!")
print(f"Output shape: {hidden_states.shape}")
print(f"Router stats: {router_stats}")
```

### 🔍 Understanding Router Statistics

Router statistics show how the model chooses between experts:

```python
# Analyze routing decisions
for layer_idx, stats in router_stats.items():
    expert_weights = stats['expert_weights']
    print(f"Layer {layer_idx}:")
    print(f"  DialoGPT weight: {expert_weights[0]:.3f}")
    print(f"  DistilGPT-2 weight: {expert_weights[1]:.3f}")
    print(f"  Entropy: {stats['entropy']:.3f}")
```

## 📝 Step 8: Generate Text

```python
# Generate text with the fused model
print("📝 Generating text...")

# Generation parameters
generation_params = {
    'max_length': 50,           # Maximum output length
    'temperature': 0.7,         # Creativity (0.1=conservative, 1.5=creative)
    'do_sample': True,          # Enable sampling
    'pad_token_id': mol.tokenizer.eos_token_id,
    'no_repeat_ngram_size': 2   # Avoid repetition
}

# Generate
generated = mol.generate(inputs['input_ids'], **generation_params)

# Decode and display
output_text = mol.tokenizer.decode(generated[0], skip_special_tokens=True)
print(f"🎯 Generated text: {output_text}")
```

## 📊 Step 9: Compare with Individual Models

Let's see how the fusion compares to individual models:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_individual_model(model_name, input_text):
    """Test an individual model for comparison."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    inputs = tokenizer(input_text, return_tensors="pt")
    generated = model.generate(
        inputs['input_ids'],
        max_length=50,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(generated[0], skip_special_tokens=True)

# Test both individual models
print("\n🔍 Comparison with individual models:")
print(f"DialoGPT-small: {test_individual_model('microsoft/DialoGPT-small', test_text)}")
print(f"DistilGPT-2: {test_individual_model('distilgpt2', test_text)}")
print(f"MoL Fusion: {output_text}")
```

## 🎨 Step 10: Experiment with Different Inputs

Try different types of inputs to see how routing adapts:

```python
test_cases = [
    "Hello, how are you?",                    # Conversational
    "The quick brown fox jumps",              # General text
    "What is artificial intelligence?",        # Question
    "Once upon a time in a distant land",     # Story beginning
    "How to bake a chocolate cake:",          # Instructional
]

print("\n🎨 Testing different input types:")
for test_input in test_cases:
    inputs = mol.tokenizer(test_input, return_tensors="pt")
    hidden_states, router_stats = mol.forward(
        inputs['input_ids'], 
        inputs['attention_mask'],
        return_router_stats=True
    )
    
    # Show routing preference
    layer_0_weights = router_stats[0]['expert_weights']
    dialgo_weight = layer_0_weights[0]
    distil_weight = layer_0_weights[1]
    
    preference = "DialoGPT" if dialgo_weight > distil_weight else "DistilGPT-2"
    confidence = max(dialgo_weight, distil_weight)
    
    print(f"'{test_input[:30]}...' → {preference} ({confidence:.2f})")
```

## 💾 Step 11: Save Your Model

```python
# Save the fused model
save_path = "./my_first_fusion"
print(f"💾 Saving model to {save_path}...")

mol.save_checkpoint(save_path, use_safetensors=True)
print("✅ Model saved successfully!")

# Verify by loading
print("🔄 Testing load...")
loaded_mol = MoLRuntime.load_checkpoint(save_path)
print("✅ Model loaded successfully!")
```

## 🎓 Complete Example

Here's the complete code for easy copy-paste:

```python
from mol import MoLRuntime, MoLConfig
import torch

# Configuration
config = MoLConfig(
    models=["microsoft/DialoGPT-small", "distilgpt2"],
    adapter_type="linear",
    router_type="simple",
    max_layers=2,
    memory_efficient=True
)

# Initialize and setup
mol = MoLRuntime(config)
mol.setup_embeddings()
mol.setup_lm_head()

# Add fusion layers
mol.add_layer([("microsoft/DialoGPT-small", 0), ("distilgpt2", 0)], layer_idx=0)
mol.add_layer([("microsoft/DialoGPT-small", 1), ("distilgpt2", 1)], layer_idx=1)

# Test generation
inputs = mol.tokenizer("Hello, how are you?", return_tensors="pt")
generated = mol.generate(inputs['input_ids'], max_length=30, temperature=0.7)
output = mol.tokenizer.decode(generated[0], skip_special_tokens=True)

print(f"Generated: {output}")
```

## 🔍 What Just Happened?

Congratulations! You've created your first MoL fusion. Here's what happened:

1. **Model Loading**: Two models were loaded and analyzed
2. **Dimension Matching**: Adapters ensured compatibility between different hidden dimensions
3. **Dynamic Routing**: The router learned to choose between experts based on input context
4. **Text Generation**: The fused model generated text using the best of both models

## 🎯 Key Concepts Learned

- **Dynamic Fusion**: Unlike static merging, MoL preserves both models and routes dynamically
- **Adapters**: Handle dimension mismatches between models  
- **Routers**: Make intelligent choices about which expert to use
- **Expert Selection**: Each model layer becomes an "expert" in the fusion

## 🚀 Next Steps

Now that you've mastered basic fusion, try:

1. **[Understanding Adapters](adapters.md)** - Learn about different adapter types
2. **[Router Basics](routers.md)** - Explore routing strategies  
3. **[Training Tutorial](training.md)** - Fine-tune your fusion
4. **[Large Models](large-models.md)** - Scale to bigger models

## 🔧 Troubleshooting

### Common Issues

**Out of Memory**:
```python
config.max_layers = 1  # Reduce layers
config.memory_efficient = True  # Enable optimizations
```

**Slow Generation**:
```python
config.router_type = "simple"  # Use pooled routing
generation_params['do_sample'] = False  # Use greedy decoding
```

**Poor Quality Output**:
```python
config.temperature = 0.5  # More focused routing
generation_params['temperature'] = 1.0  # More diverse generation
```

### Getting Help

- 📖 Check the [API Reference](../api/) for parameter details
- 💬 Ask questions in [GitHub Discussions](https://github.com/your-username/project-mango/discussions)
- 🐛 Report issues on [GitHub Issues](https://github.com/your-username/project-mango/issues)

---

**Great job completing your first fusion!** 🎉  
Ready for the next challenge? Try [Understanding Adapters](adapters.md)!