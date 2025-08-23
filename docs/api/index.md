# API Reference 📚

Complete API documentation for Project Mango's MoL system components.

## 🏗️ Core Modules

### [`mol.core`](core/) - Core Components
- [`MoLRuntime`](core/mol-runtime.md) - Main orchestration class
- [`MoLConfig`](core/mol-config.md) - Configuration management
- [`Adapters`](core/adapters.md) - Dimension matching components
- [`Routers`](core/routers.md) - Expert selection components
- [`BlockExtractor`](core/block-extractor.md) - Model layer extraction
- [`UniversalArchitecture`](core/universal-architecture.md) - Architecture handling

### [`mol.merge_methods`](merge-methods/) - Model Merging
- [`BaseMerge`](merge-methods/base-merge.md) - Base merging interface
- [`SlerpMerge`](merge-methods/slerp.md) - Spherical linear interpolation
- [`TiesMerge`](merge-methods/ties.md) - TIES merging method
- [`TaskArithmetic`](merge-methods/task-arithmetic.md) - Task arithmetic merging
- [`LinearMerge`](merge-methods/linear.md) - Linear weighted merging

### [`mol.config`](config/) - Configuration System
- [`MergeConfig`](config/merge-config.md) - Merge configuration
- [`ConfigParser`](config/config-parser.md) - YAML configuration parsing
- [`Validation`](config/validation.md) - Configuration validation

### [`mol.training`](training/) - Training Pipeline
- [`MoLTrainer`](training/trainer.md) - Main training class
- [`TrainingConfig`](training/config.md) - Training configuration

### [`mol.utils`](utils/) - Utility Functions
- [`MemoryUtils`](utils/memory-utils.md) - Memory management
- [`ModelUtils`](utils/model-utils.md) - Model utilities
- [`SafeTensorsUtils`](utils/safetensors-utils.md) - SafeTensors integration
- [`HfUtils`](utils/hf-utils.md) - Hugging Face utilities

### [`mol.cli`](cli/) - Command Line Interface
- [`MergeCLI`](cli/merge-cli.md) - Model merging CLI
- [`ValidateCLI`](cli/validate-cli.md) - Configuration validation CLI
- [`ExamplesCLI`](cli/examples-cli.md) - Example generation CLI

## 🚀 Quick Reference

### Common Imports

```python
# Core components
from mol import MoLRuntime, MoLConfig
from mol.core.adapters import LinearAdapter, BottleneckAdapter
from mol.core.routers import SimpleRouter, TokenLevelRouter

# Merging components
from mol.merge_methods import SlerpMerge, TiesMerge, LinearMerge
from mol.config import MergeConfig, ConfigParser

# Training components
from mol.training import MoLTrainer, TrainingConfig

# Utilities
from mol.utils import MemoryManager, ModelUtils
```

### Basic Usage Patterns

#### Dynamic Fusion
```python
config = MoLConfig(
    models=["gpt2", "distilgpt2"],
    adapter_type="linear",
    router_type="simple"
)
mol = MoLRuntime(config)
mol.setup_embeddings()
mol.setup_lm_head()
mol.add_layer([("gpt2", 0), ("distilgpt2", 0)], layer_idx=0)
```

#### Model Merging
```python
config = MergeConfig(
    method="slerp",
    models=["gpt2", "distilgpt2"],
    parameters={"t": 0.5}
)
merge = SlerpMerge(config)
merged_model = merge.merge(merge.load_models())
```

#### Training
```python
training_config = TrainingConfig(
    learning_rate=1e-4,
    batch_size=8,
    max_epochs=5
)
trainer = MoLTrainer(mol, training_config)
trainer.train(train_dataloader)
```

## 📖 Module Details

Click on any module above to see detailed API documentation including:

- **Class definitions** with inheritance hierarchy
- **Method signatures** with parameter types and return values
- **Usage examples** with working code
- **Configuration options** with default values
- **Error handling** and exception types
- **Performance notes** and best practices

## 🔍 Search & Navigation

- Use the search box to find specific functions or classes
- Browse by module using the navigation menu
- Check the [Examples](../examples/) section for practical usage
- See [Tutorials](../tutorials/) for step-by-step guides

## 🆕 Version Information

This API reference is for **Project Mango v1.0.0**. For changes between versions, see the [Changelog](../CHANGELOG.md).

---

**Need help?** Check our [Getting Started](../getting-started.md) guide or visit [GitHub Discussions](https://github.com/your-username/project-mango/discussions).