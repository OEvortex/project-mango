# Getting Started 🚀

Welcome to Project Mango! This guide will help you get up and running with the MoL (Modular Layer) system in just a few minutes.

> **⚠️ Beta Testing**: This project is currently in beta testing. Install from source and help us improve by reporting issues!

## 📋 Prerequisites

Before you begin, ensure you have:

- **Python 3.8+** installed
- **CUDA-compatible GPU** (optional, but recommended for better performance)
- **8GB+ RAM** (16GB+ recommended for larger models)
- **Git** for cloning the repository

## 🔧 Installation

## 🔧 Installation

### Beta Testing Installation (Recommended)

Since Project Mango is currently in beta testing, install directly from the GitHub repository:

```bash
# Clone the repository
git clone https://github.com/OEvortex/project-mango.git
cd project-mango

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Project Mango in development mode
pip install -e .
```

### Future PyPI Installation (Coming Soon)

```bash
# This will be available after beta testing
pip install project-mango-mol
```

### 🛠️ Verify Installation

```bash
# Test the installation
python -c "from mol import MoLRuntime; print('✅ Installation successful!')"

# Check CLI tools
mol-merge --help
```

## 🚀 Your First MoL Fusion

Let's start with a simple example that fuses two small models:

### 1. Basic Model Fusion

Create a new file `my_first_fusion.py`:

```python
from mol import MoLRuntime, MoLConfig

# Configure the fusion
config = MoLConfig(
    models=[
        "microsoft/DialoGPT-small",  # ~117M parameters
        "distilgpt2",                # ~82M parameters
    ],
    adapter_type="linear",           # Simple linear adapter
    router_type="simple",           # Pooled routing strategy
    max_layers=2,                   # Limit layers for quick demo
    memory_efficient=True           # Enable memory optimizations
)

# Initialize MoL runtime
print("🔄 Initializing MoL runtime...")
mol = MoLRuntime(config)

# Setup model components
print("⚙️ Setting up embeddings and language model head...")
mol.setup_embeddings()  # Use embeddings from first model
mol.setup_lm_head()     # Use LM head from first model

# Add fusion layers
print("🔗 Adding fusion layers...")
mol.add_layer([
    ("microsoft/DialoGPT-small", 0),  # Layer 0 from DialoGPT
    ("distilgpt2", 0)                 # Layer 0 from DistilGPT-2
], layer_idx=0)

mol.add_layer([
    ("microsoft/DialoGPT-small", 1),  # Layer 1 from DialoGPT
    ("distilgpt2", 1)                 # Layer 1 from DistilGPT-2
], layer_idx=1)

# Test the fusion
print("🧪 Testing fusion with sample input...")
inputs = mol.tokenizer("Hello, how are you today?", return_tensors="pt")

# Forward pass with routing statistics
hidden_states, router_stats = mol.forward(
    inputs['input_ids'], 
    inputs['attention_mask'],
    return_router_stats=True
)

# Print routing information
print(f"✅ Fusion successful!")
print(f"📊 Router statistics: {router_stats}")

# Generate text
print("📝 Generating text...")
generated = mol.generate(
    inputs['input_ids'],
    max_length=30,
    temperature=0.7,
    pad_token_id=mol.tokenizer.eos_token_id
)

output_text = mol.tokenizer.decode(generated[0], skip_special_tokens=True)
print(f"🎯 Generated text: {output_text}")
```

Run your first fusion:

```bash
python my_first_fusion.py
```

### 2. Model Merging (MergeKit-Style)

For static model merging instead of dynamic fusion:

```python
from mol import SlerpMerge
from mol.config import MergeConfig

# Configure SLERP merge
config = MergeConfig(
    method="slerp",
    models=["gpt2", "distilgpt2"],
    parameters={"t": 0.5},  # 50-50 interpolation
    output_path="./my_merged_model",
    dtype="float16"
)

# Perform the merge
print("🔄 Starting SLERP merge...")
slerp_merge = SlerpMerge(config)
models = slerp_merge.load_models()
merged_model = slerp_merge.merge(models)
slerp_merge.save_merged_model(merged_model)

print("✅ Merge complete! Model saved to ./my_merged_model")
```

### 3. Using the CLI

The CLI provides a quick way to perform merges:

```bash
# Generate example configurations
mol-merge examples ./configs

# View the generated config
cat ./configs/slerp_example.yml

# Perform a merge
mol-merge ./configs/slerp_example.yml ./output_model --device cuda --verbose
```

## 🎯 Key Concepts

### 🔄 Dynamic Fusion vs Static Merging

**Dynamic Fusion (MoL Runtime)**:
- Preserves multiple model experts
- Routes inputs to appropriate models at runtime
- Enables fine-tuning of routing decisions
- Better for diverse input types

**Static Merging**:
- Combines model weights permanently
- Smaller final model size
- Faster inference (no routing overhead)
- Better for uniform input types

### 🧭 Router Types

- **Simple Router**: Uses pooled representations for routing decisions
- **Token-Level Router**: Makes routing decisions for each token individually

### ⚙️ Adapter Types

- **Linear Adapter**: Simple linear transformation for dimension matching
- **Bottleneck Adapter**: Compressed representation with bottleneck layers

## 🎨 Working with Different Architectures

MoL supports mixing different model architectures:

```python
# Mix encoder and decoder models
config = MoLConfig(
    models=[
        "gpt2",                    # Decoder-only
        "bert-base-uncased",       # Encoder-only
        "t5-small"                 # Encoder-decoder
    ],
    adapter_type="linear",
    router_type="simple"
)
```

## 🔧 Configuration Options

### Memory Management

```python
config = MoLConfig(
    # ... other options ...
    memory_efficient=True,      # Enable memory optimizations
    offload_to_cpu=True,       # Offload unused models to CPU
    lazy_loading=True,         # Load models only when needed
    max_memory_gb=8           # Memory limit
)
```

### Training Settings

```python
from mol.training import TrainingConfig

training_config = TrainingConfig(
    learning_rate=1e-4,
    router_learning_rate=1e-5,  # Separate LR for routers
    batch_size=8,
    max_epochs=5,
    freeze_experts=True,        # Only train adapters/routers
    use_safetensors=True       # Use SafeTensors format
)
```

## 🚀 Next Steps

Now that you've completed your first fusion, here are some next steps:

### 📖 Tutorials
- [**Basic Layer Fusion**](tutorials/basic-fusion.md) - Detailed walkthrough
- [**Model Merging**](tutorials/model-merging.md) - Advanced merging techniques
- [**Training Adapters**](tutorials/training.md) - Fine-tuning your fusions
- [**Working with Large Models**](tutorials/large-models.md) - Memory optimization

### 💡 Examples
- [**Text Generation**](examples/text-generation.md) - Creative writing applications
- [**Question Answering**](examples/question-answering.md) - QA system fusion
- [**Code Generation**](examples/code-generation.md) - Programming assistance
- [**Multi-Domain Chat**](examples/multi-domain-chat.md) - Specialized chatbots

### 🔧 Advanced Topics
- [**Performance Optimization**](advanced/performance.md) - Speed and memory tips
- [**Custom Components**](advanced/custom-components.md) - Build your own adapters/routers
- [**Distributed Training**](advanced/distributed.md) - Scale across multiple GPUs
- [**Production Deployment**](advanced/deployment.md) - Deploy your fusions

## 🛠️ Troubleshooting

### Common Issues

**Out of Memory Error**:
```python
# Enable memory optimizations
config.memory_efficient = True
config.offload_to_cpu = True
config.max_layers = 2  # Reduce layers
```

**CUDA Not Available**:
```python
# Force CPU usage
config.device = "cpu"
```

**Model Loading Errors**:
```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Getting Help

- 📖 Check the [API Reference](api/) for detailed function documentation
- 💬 Visit our [GitHub Discussions](https://github.com/your-username/project-mango/discussions) for community support
- 🐛 Report bugs on our [GitHub Issues](https://github.com/your-username/project-mango/issues) page
- 📧 Email us at support@project-mango.dev

## 🎉 What's Next?

Congratulations! You've successfully:
- ✅ Installed Project Mango
- ✅ Created your first model fusion
- ✅ Learned key concepts
- ✅ Explored configuration options

Ready to dive deeper? Check out our [comprehensive tutorials](tutorials/) or explore the [API reference](api/) for advanced usage.

---

**Happy fusing! 🥭**