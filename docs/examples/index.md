# Examples 💡

Practical code examples and demonstrations of Project Mango's MoL system capabilities.

## 🎯 Quick Navigation

### 🚀 Basic Examples
- [**Model Fusion Basics**](#model-fusion-basics) - Simple two-model fusion
- [**Model Merging**](#model-merging) - Static model merging with YAML
- [**Text Generation**](#text-generation) - Creative text generation
- [**Training Example**](#training-example) - Fine-tuning adapters and routers

### 🔧 Advanced Examples  
- [**Large Model Fusion**](#large-model-fusion) - Memory-efficient large model handling
- [**Multi-Architecture Fusion**](#multi-architecture-fusion) - Combining different architectures
- [**Custom Components**](#custom-components) - Building custom adapters and routers
- [**Production Deployment**](#production-deployment) - Real-world deployment scenarios

### 🌐 Integration Examples
- [**Hugging Face Hub**](#hugging-face-hub) - Upload and download models
- [**SafeTensors Usage**](#safetensors-usage) - Secure model serialization
- [**CLI Workflows**](#cli-workflows) - Command-line automation
- [**Distributed Training**](#distributed-training) - Multi-GPU training

---

## 🚀 Basic Examples

### Model Fusion Basics

Create a simple fusion of two conversational models:

```python
from mol import MoLRuntime, MoLConfig

# Quick setup
config = MoLConfig(
    models=["microsoft/DialoGPT-small", "microsoft/DialoGPT-medium"],
    adapter_type="linear",
    router_type="simple",
    max_layers=3
)

mol = MoLRuntime(config)
mol.setup_embeddings()
mol.setup_lm_head()

# Add fusion layers
for i in range(3):
    mol.add_layer([
        ("microsoft/DialoGPT-small", i),
        ("microsoft/DialoGPT-medium", i)
    ], layer_idx=i)

# Generate text
inputs = mol.tokenizer("Hello! How can I help you today?", return_tensors="pt")
generated = mol.generate(inputs['input_ids'], max_length=50, temperature=0.8)
print(mol.tokenizer.decode(generated[0], skip_special_tokens=True))
```

### Model Merging

Static model merging using YAML configuration:

```python
from mol.merge_methods import SlerpMerge
from mol.config import MergeConfig

# SLERP merge configuration
config = MergeConfig(
    method="slerp",
    models=["gpt2", "distilgpt2"],
    parameters={"t": 0.6},  # 60% GPT-2, 40% DistilGPT-2
    output_path="./slerp_merged_model",
    dtype="float16"
)

# Perform merge
slerp = SlerpMerge(config)
models = slerp.load_models()
merged_model = slerp.merge(models)
slerp.save_merged_model(merged_model)

print("✅ Model merged successfully!")
```

**YAML Configuration Example:**
```yaml
# config.yml
merge_method: slerp
models:
  - gpt2
  - distilgpt2
parameters:
  t: 0.6
dtype: float16
output_path: ./slerp_merged
```

**CLI Usage:**
```bash
mol-merge config.yml ./output_model --device cuda --verbose
```

### Text Generation

Creative text generation with different fusion strategies:

```python
from mol import MoLRuntime, MoLConfig

# Creative writing fusion
config = MoLConfig(
    models=["gpt2", "gpt2-medium"],
    adapter_type="bottleneck",
    router_type="token_level",  # Per-token routing for creativity
    temperature=1.2,            # More diverse routing
    max_layers=4
)

mol = MoLRuntime(config)
mol.setup_embeddings()
mol.setup_lm_head()

# Add layers
for i in range(4):
    mol.add_layer([("gpt2", i), ("gpt2-medium", i)], layer_idx=i)

# Creative prompts
prompts = [
    "Once upon a time in a magical forest,",
    "The future of artificial intelligence is",
    "In the year 2050, humans discovered that",
    "The secret to happiness lies in"
]

for prompt in prompts:
    inputs = mol.tokenizer(prompt, return_tensors="pt")
    generated = mol.generate(
        inputs['input_ids'],
        max_length=100,
        temperature=0.9,
        top_p=0.9,
        do_sample=True
    )
    
    output = mol.tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Generated: {output}")
    print("-" * 50)
```

### Training Example

Fine-tune adapters and routers for better performance:

```python
from mol import MoLRuntime, MoLConfig
from mol.training import MoLTrainer, TrainingConfig
from datasets import load_dataset
from torch.utils.data import DataLoader

# Setup model
config = MoLConfig(
    models=["distilgpt2", "microsoft/DialoGPT-small"],
    adapter_type="bottleneck",
    router_type="simple",
    max_layers=3
)

mol = MoLRuntime(config)
mol.setup_embeddings()
mol.setup_lm_head()

# Add layers
for i in range(3):
    mol.add_layer([("distilgpt2", i), ("microsoft/DialoGPT-small", i)], layer_idx=i)

# Prepare dataset
def prepare_data():
    dataset = load_dataset("wikitext", "wikitext-2-v1", split="train[:1000]")
    
    def tokenize_function(examples):
        return mol.tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return DataLoader(tokenized_dataset, batch_size=4, shuffle=True)

# Training configuration
training_config = TrainingConfig(
    learning_rate=1e-4,
    router_learning_rate=1e-5,  # Slower for routers
    batch_size=4,
    max_epochs=3,
    entropy_penalty_coeff=0.1,
    load_balancing_coeff=0.01,
    freeze_experts=True,        # Only train adapters/routers
    output_dir="./mol_training",
    use_wandb=False
)

# Train
trainer = MoLTrainer(mol, training_config)
train_dataloader = prepare_data()

trainer.train(train_dataloader)
print("✅ Training complete!")
```

---

## 🔧 Advanced Examples

### Large Model Fusion

Handle large models with memory optimization:

```python
from mol import MoLRuntime, MoLConfig

# Memory-efficient configuration for large models
config = MoLConfig(
    models=["gpt2-large", "gpt2-xl"],
    adapter_type="bottleneck",
    router_type="simple",
    max_layers=6,                    # Limit layers
    memory_efficient=True,
    use_gradient_checkpointing=True,
    device_map={                     # Distribute across GPUs
        "gpt2-large": "cuda:0",
        "gpt2-xl": "cuda:1"
    }
)

mol = MoLRuntime(config)
mol.setup_embeddings()
mol.setup_lm_head()

# Add fewer layers to manage memory
for i in range(3):  # Only use first 3 layers
    mol.add_layer([("gpt2-large", i), ("gpt2-xl", i)], layer_idx=i)

# Memory-efficient generation
inputs = mol.tokenizer("The future of AI research", return_tensors="pt")
with torch.cuda.amp.autocast():  # Mixed precision
    generated = mol.generate(inputs['input_ids'], max_length=100)

print(mol.tokenizer.decode(generated[0], skip_special_tokens=True))
```

### Multi-Architecture Fusion

Combine different model architectures:

```python
from mol import MoLRuntime, MoLConfig

# Mix encoder and decoder models
config = MoLConfig(
    models=[
        "gpt2",                    # Decoder-only (causal LM)
        "bert-base-uncased",       # Encoder-only (bidirectional)
        "t5-small"                 # Encoder-decoder
    ],
    adapter_type="linear",
    router_type="simple",
    target_hidden_dim=768,         # Common dimension
    max_layers=2
)

mol = MoLRuntime(config)
mol.setup_embeddings()  # Uses GPT-2 embeddings
mol.setup_lm_head()     # Uses GPT-2 LM head

# Add fusion layers
mol.add_layer([
    ("gpt2", 0),
    ("bert-base-uncased", 0),
    ("t5-small", 0)  # Uses encoder layers
], layer_idx=0)

# Test with different input types
test_cases = [
    "Complete this sentence: The weather today is",
    "Analyze this text for sentiment: I love programming!",
    "Summarize: Machine learning is a subset of AI."
]

for text in test_cases:
    inputs = mol.tokenizer(text, return_tensors="pt")
    hidden_states, router_stats = mol.forward(
        inputs['input_ids'],
        return_router_stats=True
    )
    
    # Show which architecture was preferred
    weights = router_stats[0]['expert_weights']
    architectures = ["GPT-2", "BERT", "T5"]
    preferred = architectures[weights.argmax()]
    
    print(f"Input: {text[:40]}...")
    print(f"Preferred: {preferred} (weight: {weights.max():.3f})")
    print()
```

### Custom Components

Build custom adapters and routers:

```python
import torch
import torch.nn as nn
from mol.core.adapters import BaseAdapter
from mol.core.routers import BaseRouter

class AttentionAdapter(BaseAdapter):
    """Custom adapter using attention mechanism."""
    
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            batch_first=True
        )
        self.projection = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x):
        # Self-attention for feature refinement
        attended, _ = self.attention(x, x, x)
        
        # Project to target dimension
        projected = self.projection(attended)
        
        # Layer norm and residual (if possible)
        if self.input_dim == self.output_dim:
            return self.layer_norm(projected + x)
        else:
            return self.layer_norm(projected)

class LearnedTemperatureRouter(BaseRouter):
    """Router with learnable temperature parameter."""
    
    def __init__(self, hidden_dim, num_experts):
        super().__init__(hidden_dim, num_experts)
        
        self.router_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_experts)
        )
        
        # Learnable temperature
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, x, attention_mask=None):
        # Pool sequence
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(x).float()
            pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            pooled = x.mean(dim=1)
        
        # Router logits
        logits = self.router_mlp(pooled)
        
        # Apply learnable temperature
        logits = logits / torch.clamp(self.temperature, min=0.1, max=5.0)
        
        # Expert weights
        weights = torch.softmax(logits, dim=-1)
        weights = weights.unsqueeze(1).expand(-1, x.size(1), -1)
        
        return weights, logits

# Use custom components
def create_custom_fusion():
    from mol.core.mol_runtime import MoLConfig, MoLRuntime
    
    config = MoLConfig(
        models=["gpt2", "distilgpt2"],
        max_layers=2
    )
    
    mol = MoLRuntime(config)
    mol.setup_embeddings()
    mol.setup_lm_head()
    
    # Manually create layers with custom components
    from mol.core.block_extractor import ExtractedBlock
    from mol.core.mol_runtime import LayerSpec
    
    experts = [
        mol.block_extractor.extract_block("gpt2", 0),
        mol.block_extractor.extract_block("distilgpt2", 0)
    ]
    
    adapters = [
        AttentionAdapter(expert.input_dim, mol.target_hidden_dim)
        for expert in experts
    ]
    
    router = LearnedTemperatureRouter(mol.target_hidden_dim, len(experts))
    
    layer_spec = LayerSpec(
        experts=experts,
        adapters=adapters,
        router=router,
        layer_idx=0
    )
    
    # Add to model (would need to wrap in MoLLayer)
    print("Custom components created successfully!")

create_custom_fusion()
```

---

## 🌐 Integration Examples

### Hugging Face Hub

Upload and share your fusion models:

```python
from mol import MoLRuntime, MoLConfig

# Create and train your fusion
config = MoLConfig(models=["gpt2", "distilgpt2"])
mol = MoLRuntime(config)
# ... setup and training ...

# Push to Hugging Face Hub
mol.push_to_hf(
    repo_id="username/my-fusion-model",
    fusion_type="runtime",           # Lightweight runtime
    commit_message="Upload MoL fusion model",
    private=False
)

# Alternative: Push fully fused model
mol.push_to_hf(
    repo_id="username/my-fused-model",
    fusion_type="fused",            # Static merged model
    fusion_method="weighted_average",
    commit_message="Upload fused model"
)

# Load from Hub
loaded_mol = MoLRuntime.from_hf("username/my-fusion-model")
```

**CLI Usage:**
```bash
# Push runtime
mol-merge push-hf username/my-model \
  --mol-checkpoint ./model.pt \
  --fusion-type runtime

# Push fused model  
mol-merge push-hf username/my-fused \
  --mol-checkpoint ./model.pt \
  --fusion-type fused \
  --fusion-method weighted_average
```

### SafeTensors Usage

Secure model serialization with SafeTensors:

```python
from mol import MoLRuntime, MoLConfig
from mol.utils.safetensors_utils import save_model_safe, load_model_safe

# Enable SafeTensors by default
config = MoLConfig(
    models=["gpt2", "distilgpt2"],
    adapter_type="linear"
)

mol = MoLRuntime(config)
mol.setup_embeddings()
mol.setup_lm_head()

# ... setup layers ...

# Save with SafeTensors (default)
mol.save_checkpoint("./model", use_safetensors=True)

# Manual SafeTensors operations
metadata = {
    "model_type": "mol_fusion",
    "version": "1.0.0",
    "created_by": "Project Mango",
    "fusion_config": str(config)
}

# Save with custom metadata
save_model_safe(mol.state_dict(), "./custom_model", metadata=metadata)

# Load and inspect metadata
loaded_state, loaded_metadata = load_model_safe("./custom_model")
print(f"Model metadata: {loaded_metadata}")

# Training with SafeTensors
from mol.training import TrainingConfig

training_config = TrainingConfig(
    use_safetensors=True,  # Default
    output_dir="./training_output"
)
```

### CLI Workflows

Automate common tasks with CLI scripts:

```bash
#!/bin/bash
# fusion_pipeline.sh - Complete fusion pipeline

# 1. Generate example configurations
echo "📝 Generating configurations..."
mol-merge examples ./configs

# 2. Validate configurations
echo "✅ Validating configurations..."
for config in ./configs/*.yml; do
    mol-merge validate "$config" || exit 1
done

# 3. Perform merges
echo "🔄 Performing merges..."
mol-merge ./configs/slerp_example.yml ./models/slerp_merged --device cuda
mol-merge ./configs/ties_example.yml ./models/ties_merged --device cuda

# 4. Test models
echo "🧪 Testing merged models..."
python test_merged_models.py

# 5. Push to Hub (optional)
if [ "$PUSH_TO_HUB" = "true" ]; then
    echo "🌐 Pushing to Hugging Face Hub..."
    mol-merge push-hf username/slerp-model \
        --model-path ./models/slerp_merged \
        --fusion-type fused
fi

echo "✅ Pipeline complete!"
```

**Batch Processing Script:**
```python
# batch_fusion.py
import yaml
from mol.merge_methods import create_merge_method
from mol.config import MergeConfig

def batch_merge(config_dir, output_dir):
    """Perform batch merging from configuration directory."""
    import os
    
    for config_file in os.listdir(config_dir):
        if config_file.endswith('.yml'):
            print(f"Processing {config_file}...")
            
            # Load configuration
            with open(os.path.join(config_dir, config_file)) as f:
                config_dict = yaml.safe_load(f)
            
            config = MergeConfig(**config_dict)
            config.output_path = os.path.join(output_dir, config_file[:-4])
            
            # Perform merge
            merge_method = create_merge_method(config)
            models = merge_method.load_models()
            merged = merge_method.merge(models)
            merge_method.save_merged_model(merged)
            
            print(f"✅ {config_file} complete!")

if __name__ == "__main__":
    batch_merge("./configs", "./merged_models")
```

### Distributed Training

Scale training across multiple GPUs:

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from mol import MoLRuntime, MoLConfig
from mol.training import MoLTrainer, TrainingConfig

def setup_distributed():
    """Setup distributed training."""
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())

def create_distributed_model():
    """Create model for distributed training."""
    config = MoLConfig(
        models=["gpt2-medium", "gpt2-large"],
        adapter_type="bottleneck",
        router_type="simple",
        max_layers=6,
        memory_efficient=True,
        use_gradient_checkpointing=True
    )
    
    mol = MoLRuntime(config)
    mol.setup_embeddings()
    mol.setup_lm_head()
    
    # Add layers
    for i in range(6):
        mol.add_layer([("gpt2-medium", i), ("gpt2-large", i)], layer_idx=i)
    
    # Wrap with DDP
    mol = DDP(mol, device_ids=[dist.get_rank()])
    return mol

def distributed_training():
    """Run distributed training."""
    setup_distributed()
    
    mol = create_distributed_model()
    
    training_config = TrainingConfig(
        learning_rate=1e-4,
        batch_size=8,  # Per GPU batch size
        max_epochs=10,
        gradient_clip_norm=1.0,
        output_dir=f"./distributed_output_rank_{dist.get_rank()}",
        use_wandb=dist.get_rank() == 0  # Only log from rank 0
    )
    
    trainer = MoLTrainer(mol, training_config)
    # trainer.train(train_dataloader)  # Your dataloader here
    
    print(f"Training complete on rank {dist.get_rank()}")

# Launch with: torchrun --nproc_per_node=4 distributed_training.py
if __name__ == "__main__":
    distributed_training()
```

---

## 🎯 Example Scripts

All examples are available as standalone scripts in the [`examples/`](../../examples/) directory:

- [`basic_fusion_demo.py`](../../examples/basic_fusion_demo.py) - Basic two-model fusion
- [`comprehensive_demo.py`](../../examples/comprehensive_demo.py) - Full feature demonstration
- [`merge_demo.py`](../../examples/merge_demo.py) - Model merging examples
- [`training_example.py`](../../examples/training_example.py) - Training demonstration
- [`safetensors_demo.py`](../../examples/safetensors_demo.py) - SafeTensors integration
- [`hf_push_demo.py`](../../examples/hf_push_demo.py) - Hugging Face Hub integration
- [`universal_architecture_demo.py`](../../examples/universal_architecture_demo.py) - Multi-architecture fusion

## 🚀 Running Examples

```bash
# Basic fusion
python examples/basic_fusion_demo.py

# Comprehensive demo with training
python examples/comprehensive_demo.py --train --eval --small-models

# Model merging
python examples/merge_demo.py

# SafeTensors demo
python examples/safetensors_demo.py
```

## 📚 Related Documentation

- [**Tutorials**](../tutorials/) - Step-by-step learning guides
- [**API Reference**](../api/) - Complete function documentation
- [**Configuration**](../configuration.md) - YAML and CLI configuration
- [**Architecture**](../architecture.md) - System design overview

---

**Need more examples?** Check our [GitHub repository](https://github.com/your-username/project-mango/tree/main/examples) or ask in [Discussions](https://github.com/your-username/project-mango/discussions)!