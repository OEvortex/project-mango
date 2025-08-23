# MoLConfig API

The `MoLConfig` class defines configuration parameters for the MoL runtime system.

## Class Definition

```python
@dataclass
class MoLConfig:
    """Configuration for MoL runtime."""
```

## Parameters

### Required Parameters

#### `models: List[str]`
List of model names or paths to use in the fusion.

**Type:** `List[str]`  
**Example:**
```python
models = [
    "microsoft/DialoGPT-small",
    "distilgpt2",
    "gpt2"
]
```

### Optional Parameters

#### `adapter_type: str = "linear"`
Type of adapter to use for dimension matching.

**Options:**
- `"linear"`: Simple linear transformation
- `"bottleneck"`: Bottleneck adapter with reduced parameters

**Default:** `"linear"`

#### `router_type: str = "simple"`
Type of router to use for expert selection.

**Options:**
- `"simple"`: Pooled routing (sequence-level decisions)
- `"token_level"`: Token-level routing (per-token decisions)

**Default:** `"simple"`

#### `max_layers: int = 32`
Maximum number of layers to use from each model.

**Default:** `32`

#### `target_hidden_dim: Optional[int] = None`
Target hidden dimension for the fusion pipeline. If None, uses the largest model's dimension.

**Default:** `None`

#### `use_gradient_checkpointing: bool = False`
Whether to use gradient checkpointing to save memory during training.

**Default:** `False`

#### `memory_efficient: bool = True`
Enable memory optimizations like lazy loading and smart device placement.

**Default:** `True`

#### `temperature: float = 1.0`
Temperature for router softmax. Lower values make routing more decisive.

**Default:** `1.0`

#### `entropy_penalty_coeff: float = 0.1`
Coefficient for entropy regularization loss to encourage exploration.

**Default:** `0.1`

#### `load_balancing_coeff: float = 0.01`
Coefficient for load balancing loss to ensure even expert usage.

**Default:** `0.01`

#### `top_k_experts: Optional[int] = None`
Number of top experts to use (for sparse routing). If None, uses all experts.

**Default:** `None`

#### `device_map: Optional[Dict[str, str]] = None`
Mapping of model names to devices for distributed inference.

**Example:**
```python
device_map = {
    "gpt2": "cuda:0",
    "distilgpt2": "cuda:1",
    "bert-base-uncased": "cpu"
}
```

#### `trust_remote_code: bool = False`
Whether to trust and execute remote code from model repositories.

**Security Note:** Only enable for trusted models as this can execute arbitrary code.

**Default:** `False`

## Usage Examples

### Basic Configuration

```python
from mol import MoLConfig

config = MoLConfig(
    models=["gpt2", "distilgpt2"],
    adapter_type="linear",
    router_type="simple"
)
```

### Memory-Efficient Configuration

```python
config = MoLConfig(
    models=["gpt2-medium", "gpt2-large"],
    adapter_type="bottleneck",
    router_type="simple",
    max_layers=8,
    memory_efficient=True,
    use_gradient_checkpointing=True
)
```

### Advanced Configuration

```python
config = MoLConfig(
    models=[
        "microsoft/DialoGPT-small",
        "distilgpt2",
        "gpt2"
    ],
    adapter_type="bottleneck",
    router_type="token_level",
    max_layers=6,
    target_hidden_dim=768,
    temperature=0.8,
    entropy_penalty_coeff=0.15,
    load_balancing_coeff=0.02,
    top_k_experts=2,
    device_map={
        "microsoft/DialoGPT-small": "cuda:0",
        "distilgpt2": "cuda:0", 
        "gpt2": "cuda:1"
    },
    trust_remote_code=False
)
```

### Multi-Architecture Configuration

```python
config = MoLConfig(
    models=[
        "gpt2",                    # Decoder-only
        "bert-base-uncased",       # Encoder-only
        "t5-small"                 # Encoder-decoder
    ],
    adapter_type="linear",
    router_type="simple",
    target_hidden_dim=768,
    memory_efficient=True
)
```

## Configuration Validation

The MoL system automatically validates configuration parameters:

### Automatic Validation

```python
try:
    config = MoLConfig(
        models=[],  # Empty list - invalid!
        adapter_type="invalid_type"  # Invalid adapter type!
    )
    mol = MoLRuntime(config)
except ValueError as e:
    print(f"Configuration error: {e}")
```

### Manual Validation

```python
def validate_config(config: MoLConfig) -> bool:
    """Validate MoL configuration."""
    if not config.models:
        raise ValueError("At least one model must be specified")
    
    if config.adapter_type not in ["linear", "bottleneck"]:
        raise ValueError(f"Invalid adapter_type: {config.adapter_type}")
    
    if config.router_type not in ["simple", "token_level"]:
        raise ValueError(f"Invalid router_type: {config.router_type}")
    
    if config.max_layers <= 0:
        raise ValueError("max_layers must be positive")
    
    return True
```

## Configuration Templates

### Small Models (Fast Prototyping)

```python
SMALL_MODEL_CONFIG = MoLConfig(
    models=["distilgpt2", "microsoft/DialoGPT-small"],
    adapter_type="linear",
    router_type="simple",
    max_layers=4,
    memory_efficient=True
)
```

### Medium Models (Balanced Performance)

```python
MEDIUM_MODEL_CONFIG = MoLConfig(
    models=["gpt2", "gpt2-medium"],
    adapter_type="bottleneck",
    router_type="token_level",
    max_layers=8,
    use_gradient_checkpointing=True,
    memory_efficient=True
)
```

### Large Models (Maximum Performance)

```python
LARGE_MODEL_CONFIG = MoLConfig(
    models=["gpt2-large", "gpt2-xl"],
    adapter_type="bottleneck",
    router_type="token_level",
    max_layers=12,
    use_gradient_checkpointing=True,
    memory_efficient=True,
    top_k_experts=2,
    device_map={
        "gpt2-large": "cuda:0",
        "gpt2-xl": "cuda:1"
    }
)
```

## Parameter Tuning Guidelines

### Router Temperature
- **Higher (>1.0)**: More exploration, softer routing decisions
- **Lower (<1.0)**: More exploitation, sharper routing decisions
- **Recommended**: Start with 1.0, adjust based on task requirements

### Entropy Penalty
- **Higher values**: More exploration, balanced expert usage
- **Lower values**: More focused routing, potential expert collapse
- **Recommended**: 0.1 for most tasks, 0.2 for diverse datasets

### Load Balancing
- **Higher values**: Forces even expert usage, may hurt performance
- **Lower values**: Allows natural expert specialization
- **Recommended**: 0.01 for most tasks, 0.02 for imbalanced data

### Top-K Experts
- **None**: Use all experts (dense routing)
- **1**: Only use best expert (fastest, least expressive)
- **2-3**: Good balance of speed and expressiveness
- **Recommended**: Start with None, use 2 for efficiency

## Security Considerations

### `trust_remote_code` Parameter

```python
# Secure (recommended)
config = MoLConfig(
    models=["gpt2", "bert-base-uncased"],
    trust_remote_code=False  # Default: secure
)

# Only for trusted models
config = MoLConfig(
    models=["custom-model-with-remote-code"],
    trust_remote_code=True  # Explicit opt-in
)
```

When `trust_remote_code=True`, the system will:
- Display security warnings
- Log the models being loaded with remote code
- Require explicit user acknowledgment

## Configuration Serialization

### Save Configuration

```python
import json

config = MoLConfig(models=["gpt2", "distilgpt2"])

# Save to JSON
config_dict = config.__dict__
with open("mol_config.json", "w") as f:
    json.dump(config_dict, f, indent=2)
```

### Load Configuration

```python
import json

# Load from JSON
with open("mol_config.json", "r") as f:
    config_dict = json.load(f)

config = MoLConfig(**config_dict)
```

## See Also

- [`MoLRuntime`](mol-runtime.md) - Main runtime class that uses this configuration
- [`Adapters`](adapters.md) - Adapter implementations
- [`Routers`](routers.md) - Router implementations
- [`TrainingConfig`](../training/config.md) - Training-specific configuration