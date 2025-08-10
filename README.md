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

## Quick Start

```python
from mol import MoLRuntime, LinearAdapter, SimpleRouter

# Initialize MoL runtime with target models
mol = MoLRuntime(
    models=["microsoft/DialoGPT-small", "microsoft/DialoGPT-medium"],
    adapter_type="linear",
    router_type="simple"
)

# Load and prepare models
mol.load_models()

# Forward pass with dynamic routing
outputs = mol.forward(input_ids, attention_mask)
```

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Development Status

This project follows a phased development approach:

- ✅ **Phase 1**: Design & Infrastructure (Complete)
- 🚧 **Phase 2**: Prototype Runtime (In Progress)
- ⏳ **Phase 3**: Small-scale Training & Stabilization (Planned)
- ⏳ **Phase 4**: Fine-tuning & Specialization (Planned)
- ⏳ **Phase 5**: Evaluation & Ablation (Planned)
- ⏳ **Phase 6**: Deployment & Optimization (Planned)

## License

MIT License - see LICENSE file for details.