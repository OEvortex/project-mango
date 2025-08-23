# Configuration Guide ⚙️

Comprehensive guide to configuring Project Mango's MoL system using YAML files and CLI tools.

## 🎯 Overview

Project Mango supports multiple configuration methods:

- **🐍 Python API**: Direct configuration in code
- **📄 YAML Files**: Human-readable configuration files
- **🖥️ CLI Tools**: Command-line configuration and execution
- **🔧 Environment Variables**: Runtime environment settings

## 📄 YAML Configuration

### Basic YAML Structure

```yaml
# mol_config.yml
merge_method: slerp
models:
  - gpt2
  - distilgpt2
parameters:
  t: 0.5
dtype: float16
output_path: ./merged_model
trust_remote_code: false
device: cuda
```

### Complete YAML Schema

```yaml
# Complete configuration example
merge_method: slerp              # Required: merge method
base_model: gpt2                 # Optional: base model for certain methods

# Model configuration
models:                          # Required: list of models to merge
  - name: gpt2
    weight: 1.0                  # Optional: model weight
    layer_range: [0, 12]         # Optional: layer range to use
  - name: distilgpt2
    weight: 0.8
    layer_range: [0, 6]

# Model slices (advanced)
slices:
  - model: gpt2
    layer_range: [0, 12]
    weight: 1.0
  - model: distilgpt2
    layer_range: [0, 6]
    weight: 0.8

# Method-specific parameters
parameters:
  t: 0.5                        # SLERP interpolation factor
  density: 0.8                  # TIES density parameter
  normalize: true               # TIES normalization
  majority_sign_method: total   # TIES sign method
  rescale: true                 # Task arithmetic rescaling

# Output configuration
output_path: ./merged_model      # Required: output directory
dtype: float16                   # Optional: model precision
trust_remote_code: false        # Security: allow remote code
device: cuda                     # Computation device

# Advanced options
memory_efficient: true          # Enable memory optimizations
use_safetensors: true           # Use SafeTensors format
push_to_hub: false              # Auto-push to HF Hub
repo_id: username/model-name    # HF Hub repository ID
```

### Method-Specific Configurations

#### SLERP (Spherical Linear Interpolation)

```yaml
merge_method: slerp
models:
  - gpt2
  - distilgpt2
parameters:
  t: 0.5                        # Interpolation factor (0.0-1.0)
output_path: ./slerp_merged
```

#### TIES (Trim, Elect Sign, Disjoint Merge)

```yaml
merge_method: ties
base_model: gpt2                # Base model for comparison
slices:
  - model: gpt2
    layer_range: [0, 12]
    weight: 1.0
  - model: distilgpt2
    layer_range: [0, 6]
    weight: 0.8
parameters:
  density: 0.8                  # Parameter density (0.0-1.0)
  normalize: true               # Normalize weights
  majority_sign_method: total   # Sign election method
output_path: ./ties_merged
```

#### Task Arithmetic

```yaml
merge_method: task_arithmetic
base_model: gpt2                # Base model
models:
  - name: gpt2-finetuned-task1
    weight: 1.0
  - name: gpt2-finetuned-task2
    weight: 0.5
parameters:
  rescale: true                 # Rescale task vectors
  normalize: false              # Normalize vectors
output_path: ./task_arithmetic_merged
```

#### Linear Weighted Average

```yaml
merge_method: linear
models:
  - name: gpt2
    weight: 0.6
  - name: distilgpt2
    weight: 0.4
parameters:
  normalize_weights: true       # Normalize weights to sum to 1
output_path: ./linear_merged
```

## 🖥️ CLI Tools

### Main CLI Commands

#### `mol-merge` - Model Merging

```bash
# Basic usage
mol-merge config.yml output_dir

# With options
mol-merge config.yml output_dir \
  --device cuda \
  --verbose \
  --dtype float16 \
  --trust-remote-code
```

**Options:**
- `--device`: Computation device (cuda, cpu, auto)
- `--verbose`: Enable verbose logging
- `--dtype`: Model precision (float32, float16, bfloat16)
- `--trust-remote-code`: Allow remote code execution
- `--no-safetensors`: Disable SafeTensors format
- `--push-to-hub`: Auto-push to Hugging Face Hub

#### `mol-merge validate` - Configuration Validation

```bash
# Validate single configuration
mol-merge validate config.yml

# Validate multiple configurations
mol-merge validate configs/*.yml

# Strict validation with warnings as errors
mol-merge validate config.yml --strict
```

#### `mol-merge examples` - Generate Example Configurations

```bash
# Generate all example configurations
mol-merge examples ./configs

# Generate specific method examples
mol-merge examples ./configs --method slerp
mol-merge examples ./configs --method ties

# Generate with custom models
mol-merge examples ./configs --models gpt2,distilgpt2
```

#### `mol-merge push-hf` - Hugging Face Hub Integration

```bash
# Push merged model
mol-merge push-hf username/model-name \
  --model-path ./merged_model \
  --fusion-type fused

# Push MoL runtime
mol-merge push-hf username/mol-model \
  --mol-checkpoint ./model.pt \
  --fusion-type runtime
```

### CLI Configuration Files

#### Global Configuration

Create `~/.mol/config.yaml` for global settings:

```yaml
# Global MoL configuration
default_device: cuda
default_dtype: float16
cache_dir: ~/.mol/cache
hf_token: your_huggingface_token
wandb_api_key: your_wandb_key

# Default merge settings
default_merge_method: slerp
use_safetensors: true
memory_efficient: true

# Logging
log_level: INFO
log_file: ~/.mol/mol.log
```

#### Project Configuration

Create `mol.yaml` in your project root:

```yaml
# Project-specific configuration
project_name: my_mol_project
output_dir: ./outputs
models_dir: ./models

# Default parameters for this project
default_models:
  - gpt2
  - distilgpt2

default_parameters:
  slerp:
    t: 0.5
  ties:
    density: 0.8
    normalize: true
  linear:
    normalize_weights: true
```

## 🔧 Environment Variables

Control runtime behavior with environment variables:

```bash
# Device and performance
export MOL_DEVICE=cuda           # Default device
export MOL_DTYPE=float16         # Default precision
export MOL_MEMORY_EFFICIENT=true # Memory optimizations

# Caching and storage
export MOL_CACHE_DIR=~/.mol/cache
export MOL_USE_SAFETENSORS=true

# Security
export MOL_TRUST_REMOTE_CODE=false
export MOL_ALLOW_PATTERNS="*.json,*.txt"

# Hugging Face integration
export HF_TOKEN=your_token
export HF_DATASETS_CACHE=~/.mol/datasets

# Weights & Biases
export WANDB_API_KEY=your_key
export WANDB_PROJECT=mol_experiments

# Logging
export MOL_LOG_LEVEL=INFO
export MOL_VERBOSE=true
```

## 🎨 Configuration Templates

### Small Models (Fast Prototyping)

```yaml
# small_models.yml
merge_method: slerp
models:
  - distilgpt2
  - microsoft/DialoGPT-small
parameters:
  t: 0.5
dtype: float16
memory_efficient: true
output_path: ./small_merged
```

### Medium Models (Balanced Performance)

```yaml
# medium_models.yml
merge_method: ties
base_model: gpt2
slices:
  - model: gpt2
    layer_range: [0, 12]
    weight: 1.0
  - model: gpt2-medium
    layer_range: [0, 24]
    weight: 0.8
parameters:
  density: 0.8
  normalize: true
dtype: float16
memory_efficient: true
output_path: ./medium_merged
```

### Large Models (Maximum Performance)

```yaml
# large_models.yml
merge_method: linear
models:
  - name: gpt2-large
    weight: 0.6
  - name: gpt2-xl
    weight: 0.4
parameters:
  normalize_weights: true
dtype: float16
memory_efficient: true
use_safetensors: true
device: cuda
output_path: ./large_merged
```

### Multi-Architecture Fusion

```yaml
# multi_arch.yml
merge_method: linear
models:
  - name: gpt2
    weight: 0.4
    type: decoder
  - name: bert-base-uncased
    weight: 0.3
    type: encoder
  - name: t5-small
    weight: 0.3
    type: encoder_decoder
parameters:
  normalize_weights: true
target_hidden_dim: 768
output_path: ./multi_arch_merged
```

## 🔍 Configuration Validation

### Automatic Validation

The system automatically validates configurations:

```python
from mol.config import MergeConfig, validate_config

# Load and validate
config = MergeConfig.from_yaml("config.yml")
errors = validate_config(config)

if errors:
    for error in errors:
        print(f"❌ {error}")
else:
    print("✅ Configuration valid")
```

### Common Validation Errors

#### Missing Required Fields

```yaml
# ❌ Invalid: missing merge_method
models:
  - gpt2
  - distilgpt2
```

```yaml
# ✅ Valid: includes merge_method
merge_method: slerp
models:
  - gpt2
  - distilgpt2
```

#### Invalid Parameters

```yaml
# ❌ Invalid: t outside valid range
merge_method: slerp
parameters:
  t: 1.5  # Must be 0.0-1.0
```

```yaml
# ✅ Valid: t in valid range
merge_method: slerp
parameters:
  t: 0.5
```

#### Model Compatibility Issues

```yaml
# ❌ Invalid: incompatible architectures without adapter
merge_method: slerp
models:
  - gpt2        # 768 hidden dim
  - bert-large  # 1024 hidden dim
```

```yaml
# ✅ Valid: specify target dimension
merge_method: slerp
models:
  - gpt2
  - bert-large
target_hidden_dim: 768
```

## 📊 Configuration Best Practices

### 🎯 Performance Optimization

```yaml
# Optimized configuration
merge_method: slerp
models:
  - gpt2
  - distilgpt2
parameters:
  t: 0.5
dtype: float16           # Use half precision
memory_efficient: true  # Enable memory optimizations
use_safetensors: true   # Faster loading
device: cuda            # Use GPU
```

### 🔒 Security Best Practices

```yaml
# Secure configuration
merge_method: slerp
models:
  - gpt2                    # Use trusted models
  - distilgpt2
trust_remote_code: false   # Disable remote code
use_safetensors: true      # Secure serialization
validate_models: true      # Validate model integrity
```

### 🎨 Experiment Tracking

```yaml
# Experiment configuration
merge_method: slerp
models:
  - gpt2
  - distilgpt2
parameters:
  t: 0.5

# Experiment metadata
experiment:
  name: slerp_fusion_v1
  description: Testing SLERP fusion with GPT models
  tags: [fusion, gpt, slerp]
  
# Logging
wandb:
  project: mol_experiments
  entity: your_username
  tags: [slerp, gpt2]
```

## 🔧 Advanced Configuration

### Conditional Configuration

```yaml
# Environment-specific configuration
merge_method: slerp
models:
  - gpt2
  - distilgpt2

# Override based on environment
production:
  dtype: float16
  memory_efficient: true
  device: cuda

development:
  dtype: float32
  memory_efficient: false
  device: cpu
  verbose: true

testing:
  models:
    - distilgpt2
    - microsoft/DialoGPT-small
  max_layers: 2
```

### Dynamic Configuration

```python
# Python-based dynamic configuration
from mol.config import MergeConfig
import os

def create_dynamic_config():
    # Base configuration
    config = {
        "merge_method": "slerp",
        "models": ["gpt2", "distilgpt2"],
        "parameters": {"t": 0.5}
    }
    
    # Environment-specific overrides
    if os.getenv("MOL_ENV") == "production":
        config["dtype"] = "float16"
        config["memory_efficient"] = True
    
    # GPU availability
    if torch.cuda.is_available():
        config["device"] = "cuda"
    else:
        config["device"] = "cpu"
        config["dtype"] = "float32"
    
    return MergeConfig(**config)

config = create_dynamic_config()
```

## 📚 Related Documentation

- [**Getting Started**](getting-started.md) - Installation and first steps
- [**API Reference**](api/) - Programmatic configuration
- [**Examples**](examples/) - Configuration examples
- [**CLI Reference**](api/cli/) - Detailed CLI documentation

---

**Ready to configure your fusion?** Start with our [Getting Started](getting-started.md) guide or explore [Examples](examples/)!