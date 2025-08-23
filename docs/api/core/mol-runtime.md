# MoLRuntime API

The `MoLRuntime` class is the main orchestrator for dynamic layer fusion in the MoL system.

## Class Definition

```python
class MoLRuntime(nn.Module):
    """
    Main MoL Runtime for dynamic layer fusion.
    
    Combines transformer blocks from different models using adapters and routing.
    """
```

## Constructor

### `__init__(config: MoLConfig)`

Initialize a new MoL runtime instance.

**Parameters:**
- `config` (MoLConfig): Configuration object specifying models, adapters, and routing behavior

**Example:**
```python
from mol import MoLRuntime, MoLConfig

config = MoLConfig(
    models=["gpt2", "distilgpt2"],
    adapter_type="linear",
    router_type="simple",
    max_layers=4
)
mol = MoLRuntime(config)
```

## Core Methods

### `setup_embeddings()`

Setup embedding layer using the primary model's embeddings.

**Returns:** None

**Example:**
```python
mol.setup_embeddings()
```

### `setup_lm_head()`

Setup language modeling head using the primary model's LM head.

**Returns:** None

**Example:**
```python
mol.setup_lm_head()
```

### `add_layer(layer_specs: List[Tuple[str, int]], layer_idx: int)`

Add a MoL fusion layer with specified expert blocks.

**Parameters:**
- `layer_specs` (List[Tuple[str, int]]): List of (model_name, model_layer_idx) tuples
- `layer_idx` (int): Index of this layer in the MoL pipeline

**Returns:** None

**Example:**
```python
# Add layer combining layer 0 from both models
mol.add_layer([
    ("gpt2", 0),
    ("distilgpt2", 0)
], layer_idx=0)

# Add layer combining layer 1 from both models  
mol.add_layer([
    ("gpt2", 1),
    ("distilgpt2", 1)
], layer_idx=1)
```

### `forward(input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, return_router_stats: bool = False)`

Forward pass through the MoL fusion pipeline.

**Parameters:**
- `input_ids` (torch.Tensor): Input token IDs [batch_size, seq_len]
- `attention_mask` (Optional[torch.Tensor]): Attention mask [batch_size, seq_len]
- `return_router_stats` (bool): Whether to return routing statistics

**Returns:**
- `hidden_states` (torch.Tensor): Output hidden states [batch_size, seq_len, hidden_dim]
- `router_stats` (Optional[Dict]): Routing statistics if requested

**Example:**
```python
inputs = mol.tokenizer("Hello world", return_tensors="pt")
hidden_states, router_stats = mol.forward(
    inputs['input_ids'],
    inputs['attention_mask'],
    return_router_stats=True
)
```

### `generate(input_ids: torch.Tensor, **kwargs)`

Generate text using the fused model.

**Parameters:**
- `input_ids` (torch.Tensor): Input token IDs [batch_size, seq_len]
- `**kwargs`: Additional generation parameters (max_length, temperature, etc.)

**Returns:**
- `torch.Tensor`: Generated token IDs [batch_size, new_seq_len]

**Example:**
```python
inputs = mol.tokenizer("Once upon a time", return_tensors="pt")
generated = mol.generate(
    inputs['input_ids'],
    max_length=50,
    temperature=0.7,
    pad_token_id=mol.tokenizer.eos_token_id
)
output_text = mol.tokenizer.decode(generated[0], skip_special_tokens=True)
```

## Checkpoint Management

### `save_checkpoint(path: str, use_safetensors: bool = True)`

Save the MoL runtime state to disk.

**Parameters:**
- `path` (str): Directory path to save checkpoint
- `use_safetensors` (bool): Whether to use SafeTensors format (default: True)

**Returns:** None

**Example:**
```python
mol.save_checkpoint("./my_mol_model", use_safetensors=True)
```

### `load_checkpoint(path: str) -> MoLRuntime` (Class Method)

Load a MoL runtime from checkpoint.

**Parameters:**
- `path` (str): Directory path containing checkpoint

**Returns:**
- `MoLRuntime`: Loaded MoL runtime instance

**Example:**
```python
mol = MoLRuntime.load_checkpoint("./my_mol_model")
```

## Hugging Face Integration

### `push_to_hf(repo_id: str, fusion_type: str = "runtime", **kwargs)`

Push the MoL model to Hugging Face Hub.

**Parameters:**
- `repo_id` (str): Repository ID (username/model-name)
- `fusion_type` (str): Type of fusion to push ("runtime" or "fused")
- `**kwargs`: Additional push parameters

**Returns:** None

**Example:**
```python
# Push lightweight runtime
mol.push_to_hf(
    repo_id="username/my-mol-model",
    fusion_type="runtime",
    commit_message="Upload MoL runtime"
)

# Push fully fused model
mol.push_to_hf(
    repo_id="username/my-fused-model",
    fusion_type="fused",
    fusion_method="weighted_average"
)
```

## Properties

### `tokenizer`

The tokenizer used by the MoL runtime (from the primary model).

**Type:** `transformers.PreTrainedTokenizer`

### `config`

The configuration object used to initialize the runtime.

**Type:** `MoLConfig`

### `target_hidden_dim`

The target hidden dimension for the fusion pipeline.

**Type:** `int`

### `model_infos`

Information about all loaded models.

**Type:** `Dict[str, ModelInfo]`

## Usage Patterns

### Basic Fusion Setup

```python
from mol import MoLRuntime, MoLConfig

# 1. Create configuration
config = MoLConfig(
    models=["microsoft/DialoGPT-small", "distilgpt2"],
    adapter_type="linear",
    router_type="simple",
    max_layers=2,
    memory_efficient=True
)

# 2. Initialize runtime
mol = MoLRuntime(config)

# 3. Setup components
mol.setup_embeddings()
mol.setup_lm_head()

# 4. Add fusion layers
for layer_idx in range(2):
    mol.add_layer([
        ("microsoft/DialoGPT-small", layer_idx),
        ("distilgpt2", layer_idx)
    ], layer_idx=layer_idx)

# 5. Test inference
inputs = mol.tokenizer("Hello, how are you?", return_tensors="pt")
output = mol.generate(inputs['input_ids'], max_length=30)
print(mol.tokenizer.decode(output[0], skip_special_tokens=True))
```

### Memory-Efficient Setup

```python
config = MoLConfig(
    models=["gpt2-medium", "gpt2-large"],
    adapter_type="bottleneck",
    router_type="simple",
    memory_efficient=True,
    use_gradient_checkpointing=True,
    device_map={"gpt2-medium": "cuda:0", "gpt2-large": "cuda:1"}
)

mol = MoLRuntime(config)
```

### Multi-Architecture Fusion

```python
config = MoLConfig(
    models=[
        "gpt2",                    # Decoder-only
        "bert-base-uncased",       # Encoder-only
        "t5-small"                 # Encoder-decoder
    ],
    adapter_type="linear",
    router_type="token_level",
    target_hidden_dim=768
)

mol = MoLRuntime(config)
```

## Error Handling

The `MoLRuntime` class provides comprehensive error handling:

### Common Exceptions

- `ValueError`: Invalid configuration parameters
- `RuntimeError`: Runtime errors during forward pass or setup
- `MemoryError`: Insufficient memory for model loading
- `FileNotFoundError`: Missing checkpoint files

### Example Error Handling

```python
try:
    mol = MoLRuntime(config)
    mol.setup_embeddings()
    mol.setup_lm_head()
except ValueError as e:
    print(f"Configuration error: {e}")
except RuntimeError as e:
    print(f"Runtime error: {e}")
except MemoryError as e:
    print(f"Memory error: {e}")
    # Try with memory-efficient settings
    config.memory_efficient = True
    config.max_layers = 2
    mol = MoLRuntime(config)
```

## Performance Tips

1. **Memory Management**: Enable `memory_efficient=True` for large models
2. **Gradient Checkpointing**: Use `use_gradient_checkpointing=True` during training
3. **Device Mapping**: Distribute models across multiple GPUs with `device_map`
4. **Layer Limits**: Limit `max_layers` for faster initialization and reduced memory
5. **Adapter Choice**: Use `BottleneckAdapter` for parameter efficiency

## See Also

- [`MoLConfig`](mol-config.md) - Configuration options
- [`Adapters`](adapters.md) - Adapter implementations  
- [`Routers`](routers.md) - Router implementations
- [`MoLTrainer`](../training/trainer.md) - Training the runtime