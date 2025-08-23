# Project Mango - Enhanced MoL System 🥭

**Modular Layer (MoL) System for LLMs with Advanced Model Merging**

A powerful toolkit that combines:
- 🔄 **Dynamic runtime fusion** of transformer layers from different LLMs  
- 🔀 **MergeKit-style model merging** with SLERP, TIES, Task Arithmetic, and Linear methods
- ⚡ **Memory-efficient operations** with lazy loading and smart device placement
- 🛠️ **YAML configuration system** for easy merge specification
- 🖥️ **CLI interface** similar to mergekit-yaml

## 🚀 Quick Start

### Model Merging (MergeKit-style)

**Using CLI:**
```bash
# Generate example configurations
mol-merge examples ./configs

# Validate configuration
mol-merge validate config.yml

# Perform merge
mol-merge config.yml ./merged_model --device cuda --verbose
```

**Using Python API:**
```python
from mol import SlerpMerge
from mol.merge_methods.base_merge import MergeConfig

# SLERP merge
config = MergeConfig(
    method="slerp",
    models=["gpt2", "distilgpt2"],
    parameters={"t": 0.5},
    output_path="./merged_model"
)

slerp = SlerpMerge(config)
models = slerp.load_models()
merged = slerp.merge(models)
slerp.save_merged_model(merged)
```

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

### Pushing to Hugging Face Hub 🌐

**Install HuggingFace Hub (optional):**
```bash
pip install huggingface_hub
```

**Push MoL models:**
```python
# Option 1: Push lightweight MoL runtime (~50-100MB)
mol.push_to_hf(
    repo_id="your-username/mol-model",
    fusion_type="runtime",
    commit_message="Upload MoL runtime"
)

# Option 2: Push fully fused static model (~2GB)
mol.push_to_hf(
    repo_id="your-username/fused-model", 
    fusion_type="fused",
    fusion_method="weighted_average",
    commit_message="Upload fused model"
)
```

**CLI Usage:**
```bash
# Push MoL runtime
mol-merge push-hf your-username/mol-model \
  --mol-checkpoint ./model.pt \
  --fusion-type runtime

# Push fully fused model
mol-merge push-hf your-username/fused-model \
  --mol-checkpoint ./model.pt \
  --fusion-type fused \
  --fusion-method weighted_average
```

### SafeTensors Support 🔒

**Secure model serialization (recommended):**
```python
# SafeTensors enabled by default
mol.save_checkpoint("./model", use_safetensors=True)
loaded_mol = MoLRuntime.load_checkpoint("./model")

# Training with SafeTensors
training_config = TrainingConfig(
    use_safetensors=True,  # Default: True
    output_dir="./checkpoints"
)

# Manual SafeTensors operations
from mol import save_model_safe, load_model_safe
save_model_safe(model, "./model", metadata={"version": "1.0"})
metadata = load_model_safe(model, "./model")
```

**Benefits:**
- 🛡️ **Security**: No arbitrary code execution (unlike pickle)
- ⚡ **Performance**: Faster loading and memory mapping
- 🔍 **Transparency**: Inspectable file format
- ✅ **Integrity**: Built-in validation and checksums

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