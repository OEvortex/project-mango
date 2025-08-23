# Project Mango - Enhanced MoL System 🥭

**Modular Layer (MoL) System for Dynamic Fusion and Static Merging of LLMs**

Project Mango provides **two powerful approaches** for combining transformer models:

## 🔄 **FUSION** - Dynamic Runtime Combination
- **Smart routing** between multiple model experts at inference time
- **Preserves all models** as separate experts with intelligent selection
- **Adaptive behavior** - different experts for different inputs
- **Memory efficient** with lazy loading and smart device placement
- **Trainable components** - fine-tune adapters and routers

## 🔀 **MERGE** - Static Model Combination  
- **MergeKit-style merging** with SLERP, TIES, Task Arithmetic, and Linear methods
- **Creates single unified model** by combining weights permanently
- **Smaller final size** - one model instead of multiple experts
- **YAML configuration system** for complex merge specifications
- **CLI interface** similar to mergekit-yaml

### 🎯 **When to Use Which?**

| Use Case | Recommendation | Why |
|----------|----------------|-----|
| **Diverse inputs** (chat + code + science) | 🔄 **Fusion** | Dynamic expert selection adapts to input type |
| **Uniform inputs** (only chat or only code) | 🔀 **Merge** | Simpler, faster, smaller final model |
| **Experimentation** | 🔄 **Fusion** | Easy to adjust expert combinations |
| **Production deployment** | 🔀 **Merge** | Single model, predictable performance |
| **Limited memory** | 🔀 **Merge** | Smaller memory footprint |
| **Maximum flexibility** | 🔄 **Fusion** | Best of both worlds dynamically |

## 🚀 Quick Start

### 🔀 **MERGE** - Static Model Combination (MergeKit-style)

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

## 🏢 System Architecture

Project Mango implements **two complementary approaches**:

### 🔄 **FUSION Architecture** (Dynamic MoL Runtime)
The MoL system consists of:

1. **Block Extractors**: Extract transformer blocks from different model architectures
2. **Adapters**: Bridge dimensional differences between models
3. **Routers**: Decide which expert (model layer) to use for each input
4. **MoL Runtime**: Orchestrates the entire fusion process
5. **Training Pipeline**: Fine-tune adapters and routers for optimal performance

### 🔀 **MERGE Architecture** (Static Combination)
The merge system provides:

1. **Merge Methods**: SLERP, TIES, Task Arithmetic, Linear algorithms
2. **Configuration Parser**: YAML-based merge specifications
3. **Model Loader**: Universal model loading and validation
4. **Weight Combiner**: Intelligent parameter combination
5. **Output Generator**: Creates unified model files

## 📝 Examples

### 🔄 **FUSION Examples** - Dynamic Runtime

#### Running the Comprehensive Demo
```bash
# Basic inference demonstration
python examples/comprehensive_demo.py --inference-only --small-models

# Full demo with training and evaluation  
python examples/comprehensive_demo.py --train --eval --small-models

# Large model demonstration
python examples/comprehensive_demo.py --train --eval
```

#### Multi-Domain Chatbot (Dynamic Expert Selection)
```python
from mol import MoLRuntime, MoLConfig

# Configure fusion for diverse inputs
config = MoLConfig(
    models=[
        "microsoft/DialoGPT-small",    # Conversational expert
        "Salesforce/codet5-small",     # Code expert
        "allenai/scibert_scivocab_uncased"  # Science expert
    ],
    adapter_type="linear",
    router_type="token_level",  # Per-token expert selection
    max_layers=4
)

mol = MoLRuntime(config)
mol.setup_embeddings()
mol.setup_lm_head()

# Add fusion layers
for i in range(4):
    mol.add_layer([
        ("microsoft/DialoGPT-small", i),
        ("Salesforce/codet5-small", i), 
        ("allenai/scibert_scivocab_uncased", i)
    ], layer_idx=i)

# Test with different input types
test_inputs = [
    "Hello, how are you today?",                    # Conversational
    "def fibonacci(n): return",                      # Code
    "The photosynthesis process involves"             # Science
]

for text in test_inputs:
    inputs = mol.tokenizer(text, return_tensors="pt")
    hidden_states, router_stats = mol.forward(
        inputs['input_ids'], 
        inputs['attention_mask'],
        return_router_stats=True
    )
    print(f"Input: {text}")
    print(f"Expert selection: {router_stats}")
    print()
```

### 🔀 **MERGE Examples** - Static Combination

#### SLERP Merge for Balanced Models
```bash
# Using CLI
mol-merge examples ./configs
mol-merge ./configs/slerp_example.yml ./balanced_model --device cuda
```

```python
# Using Python API
from mol import SlerpMerge
from mol.config import MergeConfig

# Create balanced model from chat and instruct variants
config = MergeConfig(
    method="slerp",
    models=["microsoft/DialoGPT-medium", "microsoft/DialoGPT-large"],
    parameters={"t": 0.5},  # 50-50 balance
    output_path="./balanced_chatbot"
)

merge = SlerpMerge(config)
models = merge.load_models()
merged = merge.merge(models)
merge.save_merged_model(merged)
```

#### TIES Merge for Capability Enhancement
```python
# Combine base model with multiple fine-tuned variants
config = MergeConfig(
    method="ties",
    base_model="gpt2",
    models=[
        "gpt2-finetuned-poetry",
        "gpt2-finetuned-science", 
        "gpt2-finetuned-code"
    ],
    parameters={
        "density": 0.8,
        "normalize": True
    },
    output_path="./enhanced_gpt2"
)
```

### 🔄 **FUSION** - Dynamic Runtime Combination
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

### 🏗️ Universal Architecture Support

MoL now supports **ALL transformer architectures** available in HuggingFace Transformers (120+) with dynamic detection and secure loading:

### 🔍 Automatic Architecture Detection

- **Dynamic Discovery**: Automatically detects any transformer architecture
- **Component Mapping**: Finds layers, embeddings, and LM heads automatically
- **Architecture Classification**: Groups models into families (decoder-only, encoder-only, etc.)
- **No Manual Updates**: Handles new architectures without code changes

### 🛡️ Security First

```python
# Secure by default - no remote code execution
config = MoLConfig(
    models=["gpt2", "bert-base-uncased", "t5-small"],
    trust_remote_code=False  # Default: secure
)

# Only enable for trusted models
config_trusted = MoLConfig(
    models=["custom-model-requiring-remote-code"],
    trust_remote_code=True  # Explicit opt-in with warnings
)
```

### 🏷️ Supported Architecture Families

| Family | Examples | MoL Support |
|--------|----------|-------------|
| **Decoder-Only** | GPT-2, GPT-Neo, Llama, Mistral, Falcon, OPT, BLOOM, Qwen, Yi | ✅ Full Support |
| **Encoder-Only** | BERT, RoBERTa, DistilBERT, ELECTRA, DeBERTa, ALBERT | ✅ Full Support |
| **Encoder-Decoder** | T5, BART, Pegasus, Marian, UL2, FLAN-T5 | ✅ Full Support |
| **Vision** | ViT, DeiT, Swin, BEiT, ConvNeXt | ✅ Full Support |
| **Multimodal** | CLIP, FLAVA, LayoutLM, LXMERT | ✅ Full Support |

### 💡 Usage Examples

```python
from mol import MoLRuntime, MoLConfig

# Mix different architecture families
config = MoLConfig(
    models=[
        "gpt2",                    # Decoder-only
        "bert-base-uncased",       # Encoder-only  
        "t5-small",               # Encoder-decoder
    ],
    adapter_type="linear",
    router_type="simple"
)

mol_runtime = MoLRuntime(config)
mol_runtime.setup_embeddings()
mol_runtime.setup_lm_head()

# Add fusion layers from different architectures
mol_runtime.add_layer([
    ("gpt2", 0),
    ("bert-base-uncased", 0),
    ("t5-small", 0)
], layer_idx=0)
```

### 🔧 Architecture Handler

```python
from mol.core import UniversalArchitectureHandler

# Create handler (secure by default)
handler = UniversalArchitectureHandler(trust_remote_code=False)

# Detect any architecture
arch_info = handler.detect_architecture("microsoft/DialoGPT-medium")
print(f"Architecture: {arch_info.architecture_type}")
print(f"Family: {arch_info.architecture_family}")
print(f"Layers: {arch_info.num_layers}")
print(f"Hidden Dim: {arch_info.hidden_dim}")
print(f"Layer Path: {arch_info.layer_path}")
print(f"Supports Causal LM: {arch_info.supports_causal_lm}")
```

### ⚡ Performance Features

- **Caching**: Architecture detection results are cached
- **Lazy Loading**: Models loaded only when needed
- **Memory Efficient**: Smart device placement and memory management
- **Fast Introspection**: Quick component discovery without full model loading

### 🔒 Security Configuration

```bash
# CLI with security options
mol-merge config.yml ./output --trust-remote-code  # Explicit opt-in
mol-merge config.yml ./output                      # Secure by default

# Configuration file
merge_method: slerp
models:
  - gpt2
  - bert-base-uncased
trust_remote_code: false  # Default: secure
output_path: ./merged_model
```

## Training Adapters and Routers
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

✅ **🔄 FUSION Components**:
- Linear and Bottleneck adapters with identity initialization
- SimpleRouter (pooled) and TokenLevelRouter (per-token) with top-k routing
- Support for GPT-2, GPT-Neo, GPT-J, LLaMA, BERT, RoBERTa, DistilBERT architectures
- Memory-efficient lazy loading and offloading
- Comprehensive training pipeline with adapter/router optimization

✅ **🔀 MERGE Components**:
- SLERP (Spherical Linear Interpolation) for smooth model blending
- TIES (Trim, Elect Sign, Disjoint Merge) for advanced capability combination
- Task Arithmetic for precise capability manipulation
- Linear weighted averaging with fine-grained control
- YAML configuration system for complex merge specifications
- CLI interface with validation and batch processing

✅ **🔍 Universal Features**:
- Support for 120+ transformer architectures from Hugging Face
- SafeTensors integration for secure model serialization
- Hugging Face Hub integration for easy model sharing
- Automatic architecture detection and component mapping
- Security-first design with configurable trust levels

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

Apache License 2.0 - see [LICENSE](LICENSE) file for details.