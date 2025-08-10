# Project Mango - Modular Layer (MoL) System

A runtime system for dynamically combining transformer blocks from arbitrary Large Language Models (LLMs) using adapters and routing mechanisms.

## Overview

The MoL system enables fusion of transformer layers from different LLMs of varying sizes (e.g., 1B to 7B parameters), allowing you to:

- **Mix and Match Layers**: Combine layers from specialist and generalist models
- **Dynamic Routing**: Use intelligent routing to select appropriate layers for different inputs
- **Efficient Memory Management**: Lazy loading and offloading for large model combinations
- **Adaptive Fusion**: Train adapters to bridge dimensional differences between models

## Key Features

- 🔄 **Layer Fusion**: Combine transformer blocks from models of different architectures and sizes
- 🧠 **Smart Routing**: Token-level and pooled routing strategies for optimal layer selection
- 📏 **Dimension Adapters**: Linear and bottleneck adapters for handling size mismatches
- 🚀 **Memory Efficient**: Lazy loading, offloading, and distributed inference support
- 🎯 **Fine-tunable**: Training pipeline for adapters and routers with identity initialization
- 📊 **Evaluation Ready**: Built-in metrics and visualization for router behavior

## Architecture

The MoL system consists of:

1. **Block Extractors**: Extract transformer blocks from different model architectures
2. **Adapters**: Bridge dimensional differences between models
3. **Routers**: Decide which expert (model layer) to use for each input
4. **MoL Runtime**: Orchestrates the entire fusion process
5. **Training Pipeline**: Fine-tune adapters and routers for optimal performance

## Examples

### Running the Comprehensive Demo
```bash
# Basic inference demonstration
python examples/comprehensive_demo.py --inference-only --small-models

# Full demo with training and evaluation  
python examples/comprehensive_demo.py --train --eval --small-models

# Large model demonstration
python examples/comprehensive_demo.py --train --eval
```

### Basic Layer Fusion
```python
from mol import MoLRuntime
from mol.core.mol_runtime import MoLConfig

# Create configuration
config = MoLConfig(
    models=["microsoft/DialoGPT-small", "distilgpt2"],
    adapter_type="linear",
    router_type="simple",
    max_layers=4
)

# Initialize MoL runtime
mol = MoLRuntime(config)

# Setup embeddings and LM head
mol.setup_embeddings()
mol.setup_lm_head()

# Add fusion layers
mol.add_layer([
    ("microsoft/DialoGPT-small", 0),
    ("distilgpt2", 0)
], layer_idx=0)

# Forward pass with dynamic routing
inputs = mol.tokenizer("Hello, how are you?", return_tensors="pt")
hidden_states, router_stats = mol.forward(
    inputs['input_ids'], 
    inputs['attention_mask'],
    return_router_stats=True
)
```

### Training Adapters and Routers
```python
from mol.training.trainer import MoLTrainer, TrainingConfig

# Training configuration
train_config = TrainingConfig(
    learning_rate=1e-4,
    router_learning_rate=1e-5,
    batch_size=8,
    max_epochs=5,
    freeze_experts=True
)

# Initialize trainer
trainer = MoLTrainer(mol, train_config)

# Train the system
trainer.train(train_dataloader, eval_dataloader)
```

### Individual Component Usage
```python
# Using adapters directly
from mol.core.adapters import LinearAdapter, BottleneckAdapter

linear_adapter = LinearAdapter(512, 768, init_identity=True)
bottleneck_adapter = BottleneckAdapter(512, 768, bottleneck_dim=128)

# Using routers directly  
from mol.core.routers import SimpleRouter, TokenLevelRouter

simple_router = SimpleRouter(768, num_experts=3, pooling_type="mean")
token_router = TokenLevelRouter(768, num_experts=3, top_k=2)
```

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Development Status

This project follows a phased development approach:

- ✅ **Phase 1**: Design & Infrastructure (Complete)
- ✅ **Phase 2**: Prototype Runtime (Complete)
- ✅ **Phase 3**: Small-scale Training & Stabilization (Complete)
- 🚧 **Phase 4**: Fine-tuning & Specialization (In Progress)
- ⏳ **Phase 5**: Evaluation & Ablation (Planned)
- ⏳ **Phase 6**: Deployment & Optimization (Planned)

### Current Capabilities

✅ **Core Components**:
- Linear and Bottleneck adapters with identity initialization
- SimpleRouter (pooled) and TokenLevelRouter (per-token) with top-k routing
- Support for GPT-2, GPT-Neo, GPT-J, LLaMA, BERT, RoBERTa, DistilBERT architectures
- Memory-efficient lazy loading and offloading
- Comprehensive training pipeline with adapter/router optimization

✅ **Training Features**:
- Identity initialization for warm start
- Separate learning rates for adapters and routers  
- Router entropy and load balancing regularization
- Gradient checkpointing and memory optimization
- Wandb integration for experiment tracking

✅ **Testing & Examples**:
- Unit tests for all core components
- Basic fusion demo with real models
- Complete training example with small models

## License

MIT License - see LICENSE file for details.