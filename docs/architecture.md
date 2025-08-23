# Architecture Overview 🏗️

Deep dive into Project Mango's MoL (Modular Layer) system architecture, design patterns, and implementation details.

## 🎯 System Overview

Project Mango implements a sophisticated **Modular Layer (MoL) system** that enables dynamic fusion of transformer layers from different Large Language Models. The architecture is designed for flexibility, scalability, and efficiency.

### Core Design Principles

1. **🔄 Dynamic Routing**: Runtime selection of optimal model experts
2. **⚡ Memory Efficiency**: Lazy loading and smart device placement
3. **🧩 Modularity**: Pluggable components for adapters and routers
4. **🔒 Security**: Safe model loading with configurable trust levels
5. **📈 Scalability**: Support for distributed training and inference

## 🏗️ High-Level Architecture

```mermaid
graph TB
    subgraph "Input Layer"
        A[Input Text] --> B[Tokenizer]
        B --> C[Input Embeddings]
    end
    
    subgraph "MoL Fusion Pipeline"
        C --> D[MoL Layer 0]
        D --> E[MoL Layer 1]
        E --> F[MoL Layer N]
    end
    
    subgraph "MoL Layer Detail"
        G[Hidden States] --> H[Router]
        H --> I[Expert Selection]
        I --> J[Expert A Block]
        I --> K[Expert B Block]
        I --> L[Expert C Block]
        J --> M[Adapter A]
        K --> N[Adapter B]
        L --> O[Adapter C]
        M --> P[Weighted Combination]
        N --> P
        O --> P
    end
    
    subgraph "Output Layer"
        F --> Q[Language Model Head]
        Q --> R[Output Probabilities]
    end
    
    D -.-> G
    P --> S[Layer Output]
    S --> E
```

## 🔧 Core Components

### 1. MoL Runtime (`mol.core.mol_runtime`)

The central orchestrator that manages the entire fusion pipeline.

```mermaid
graph LR
    subgraph "MoL Runtime"
        A[Configuration] --> B[Model Loading]
        B --> C[Component Setup]
        C --> D[Layer Management]
        D --> E[Inference Pipeline]
        E --> F[Generation Interface]
    end
    
    subgraph "External Dependencies"
        G[Block Extractor]
        H[Memory Manager]
        I[Model Utils]
    end
    
    B --> G
    C --> H
    D --> I
```

**Key Responsibilities:**
- Model initialization and configuration
- Layer composition and management
- Forward pass orchestration
- Memory and device management
- Checkpoint saving/loading

### 2. Block Extractor (`mol.core.block_extractor`)

Extracts transformer blocks from various model architectures.

```mermaid
graph TB
    A[Model Repository] --> B[Architecture Detection]
    B --> C[Component Mapping]
    C --> D[Block Extraction]
    D --> E[Dimension Analysis]
    E --> F[ExtractedBlock Object]
    
    subgraph "Supported Architectures"
        G[GPT-2/GPT-Neo]
        H[BERT/RoBERTa]
        I[T5/BART]
        J[LLaMA/Mistral]
        K[Custom Models]
    end
    
    B --> G
    B --> H
    B --> I
    B --> J
    B --> K
```

**Features:**
- Universal architecture support (120+ models)
- Automatic component discovery
- Safe model loading with trust controls
- Lazy loading and memory optimization

### 3. Adapters (`mol.core.adapters`)

Handle dimensional mismatches between different models.

```mermaid
graph LR
    subgraph "Adapter Types"
        A[LinearAdapter] --> D[Output]
        B[BottleneckAdapter] --> D
        C[CustomAdapter] --> D
    end
    
    subgraph "LinearAdapter Flow"
        E[Input] --> F[Linear Projection]
        F --> G[Layer Norm]
        G --> H[Residual Connection]
        H --> I[Output]
    end
    
    subgraph "BottleneckAdapter Flow"
        J[Input] --> K[Down Projection]
        K --> L[Activation]
        L --> M[Dropout]
        M --> N[Up Projection]
        N --> O[Layer Norm]
        O --> P[Residual Connection]
        P --> Q[Output]
    end
```

**Adapter Types:**
- **Linear**: Simple linear transformation
- **Bottleneck**: Parameter-efficient with compression
- **Custom**: User-defined adapters

### 4. Routers (`mol.core.routers`)

Intelligent expert selection mechanisms.

```mermaid
graph TB
    subgraph "Router Types"
        A[SimpleRouter] --> E[Expert Weights]
        B[TokenLevelRouter] --> E
        C[CustomRouter] --> E
    end
    
    subgraph "SimpleRouter (Pooled)"
        F[Input Sequence] --> G[Sequence Pooling]
        G --> H[Router MLP]
        H --> I[Softmax]
        I --> J[Broadcast to Sequence]
    end
    
    subgraph "TokenLevelRouter"
        K[Input Sequence] --> L[Per-Token MLP]
        L --> M[Top-K Selection]
        M --> N[Sparse Weights]
    end
```

**Router Strategies:**
- **Simple (Pooled)**: One decision per sequence
- **Token-Level**: Per-token routing decisions
- **Top-K**: Sparse expert selection
- **Learned Temperature**: Adaptive decision sharpness

### 5. Universal Architecture Handler (`mol.core.universal_architecture`)

Automatic architecture detection and component mapping.

```mermaid
graph TB
    A[Model Config] --> B[Architecture Detection]
    B --> C{Architecture Type}
    
    C -->|Decoder-Only| D[GPT/LLaMA Handler]
    C -->|Encoder-Only| E[BERT/RoBERTa Handler]
    C -->|Encoder-Decoder| F[T5/BART Handler]
    C -->|Vision| G[ViT Handler]
    C -->|Multimodal| H[CLIP Handler]
    
    D --> I[Component Mapping]
    E --> I
    F --> I
    G --> I
    H --> I
    
    I --> J[Validated Architecture Info]
```

## 🔀 Model Merging Architecture

Separate from dynamic fusion, the merging system provides static model combination.

```mermaid
graph LR
    subgraph "Merge Methods"
        A[SLERP] --> F[Merged Model]
        B[TIES] --> F
        C[Task Arithmetic] --> F
        D[Linear] --> F
        E[Custom] --> F
    end
    
    subgraph "Configuration System"
        G[YAML Config] --> H[Config Parser]
        H --> I[Validation]
        I --> J[Merge Method Factory]
    end
    
    subgraph "CLI Interface"
        K[mol-merge CLI] --> L[Config Loading]
        L --> M[Batch Processing]
        M --> N[Hub Integration]
    end
    
    J --> A
    J --> B
    J --> C
    J --> D
```

## 🧠 Training Architecture

The training system optimizes adapters and routers while keeping experts frozen.

```mermaid
graph TB
    subgraph "Training Pipeline"
        A[Training Data] --> B[Data Loader]
        B --> C[MoL Forward Pass]
        C --> D[Loss Computation]
        D --> E[Backward Pass]
        E --> F[Optimizer Step]
    end
    
    subgraph "Loss Components"
        G[Language Modeling Loss]
        H[Router Entropy Loss]
        I[Load Balancing Loss]
        J[Adapter Regularization]
    end
    
    D --> G
    D --> H
    D --> I
    D --> J
    
    subgraph "Optimization Strategy"
        K[Frozen Experts] --> L[Trainable Adapters]
        K --> M[Trainable Routers]
        L --> N[Separate Learning Rates]
        M --> N
    end
```

## 💾 Memory Management Architecture

Efficient memory usage through multiple optimization strategies.

```mermaid
graph TB
    subgraph "Memory Strategies"
        A[Lazy Loading] --> E[Memory Pool]
        B[Model Offloading] --> E
        C[Gradient Checkpointing] --> E
        D[Mixed Precision] --> E
    end
    
    subgraph "Device Management"
        F[CPU Memory] --> G[Smart Placement]
        H[GPU Memory] --> G
        I[Multiple GPUs] --> G
        G --> J[Dynamic Transfer]
    end
    
    subgraph "Optimization Techniques"
        K[Reference Counting]
        L[Cache Management]
        M[Memory Profiling]
        N[Automatic Cleanup]
    end
    
    E --> K
    E --> L
    E --> M
    E --> N
```

## 🔒 Security Architecture

Multi-layered security approach for safe model loading and execution.

```mermaid
graph TB
    subgraph "Security Layers"
        A[Input Validation] --> B[Model Verification]
        B --> C[Code Execution Control]
        C --> D[Resource Limits]
        D --> E[Safe Serialization]
    end
    
    subgraph "Trust Levels"
        F[Trusted Models] --> G[Full Access]
        H[Community Models] --> I[Restricted Access]
        J[Unknown Models] --> K[Sandboxed Execution]
    end
    
    subgraph "SafeTensors Integration"
        L[No Code Execution]
        M[Fast Loading]
        N[Integrity Checks]
        O[Metadata Validation]
    end
    
    C --> F
    C --> H
    C --> J
    E --> L
    E --> M
    E --> N
    E --> O
```

## 🌐 Integration Architecture

Seamless integration with the ML ecosystem.

```mermaid
graph LR
    subgraph "Hugging Face Ecosystem"
        A[Transformers] --> E[MoL System]
        B[Datasets] --> E
        C[Hub] --> E
        D[Tokenizers] --> E
    end
    
    subgraph "Training Frameworks"
        F[PyTorch] --> E
        G[Accelerate] --> E
        H[DeepSpeed] --> E
        I[FSDP] --> E
    end
    
    subgraph "Experiment Tracking"
        J[Wandb] --> E
        K[TensorBoard] --> E
        L[MLflow] --> E
    end
    
    subgraph "Deployment"
        M[FastAPI] --> N[Production API]
        O[Gradio] --> P[Demo Interface]
        Q[Docker] --> R[Containerized Deployment]
    end
    
    E --> M
    E --> O
    E --> Q
```

## 📊 Data Flow Architecture

Detailed data flow through the MoL system.

```mermaid
sequenceDiagram
    participant U as User Input
    participant T as Tokenizer
    participant E as Embeddings
    participant ML as MoL Layer
    participant R as Router
    participant Ex as Expert Block
    participant A as Adapter
    participant LM as LM Head
    participant O as Output

    U->>T: Input text
    T->>E: Token IDs
    E->>ML: Hidden states
    
    loop For each MoL layer
        ML->>R: Hidden states + mask
        R->>Ex: Expert weights
        Ex->>A: Expert outputs
        A->>ML: Adapted outputs
        ML->>ML: Weighted combination
    end
    
    ML->>LM: Final hidden states
    LM->>O: Token probabilities
    O->>U: Generated text
```

## 🔄 Configuration Architecture

Flexible configuration system supporting multiple input methods.

```mermaid
graph TB
    subgraph "Configuration Sources"
        A[Python API] --> F[Config Merger]
        B[YAML Files] --> F
        C[Environment Variables] --> F
        D[CLI Arguments] --> F
        E[Default Values] --> F
    end
    
    subgraph "Configuration Processing"
        F --> G[Schema Validation]
        G --> H[Type Conversion]
        H --> I[Conflict Resolution]
        I --> J[Final Configuration]
    end
    
    subgraph "Configuration Types"
        K[MoL Config]
        L[Merge Config]
        M[Training Config]
        N[Deployment Config]
    end
    
    J --> K
    J --> L
    J --> M
    J --> N
```

## 🚀 Performance Architecture

Optimization strategies for speed and efficiency.

```mermaid
graph TB
    subgraph "Compute Optimizations"
        A[Mixed Precision] --> D[Performance Gains]
        B[Gradient Checkpointing] --> D
        C[Kernel Fusion] --> D
    end
    
    subgraph "Memory Optimizations"
        E[Lazy Loading] --> H[Memory Savings]
        F[Model Offloading] --> H
        G[Cache Management] --> H
    end
    
    subgraph "Parallelization"
        I[Data Parallel] --> L[Scalability]
        J[Model Parallel] --> L
        K[Pipeline Parallel] --> L
    end
    
    subgraph "Inference Optimizations"
        M[Static Graph] --> P[Inference Speed]
        N[Batching] --> P
        O[Quantization] --> P
    end
```

## 🔧 Extension Points

Architecture designed for extensibility and customization.

```mermaid
graph LR
    subgraph "Custom Components"
        A[Custom Adapters] --> E[Plugin System]
        B[Custom Routers] --> E
        C[Custom Merge Methods] --> E
        D[Custom Architectures] --> E
    end
    
    subgraph "Extension Interfaces"
        F[BaseAdapter] --> G[Implementation]
        H[BaseRouter] --> G
        I[BaseMerge] --> G
        J[BaseArchitecture] --> G
    end
    
    subgraph "Registration System"
        K[Component Registry] --> L[Dynamic Loading]
        L --> M[Automatic Discovery]
        M --> N[Configuration Integration]
    end
    
    E --> F
    E --> H
    E --> I
    E --> J
    G --> K
```

## 📈 Monitoring and Observability

Built-in monitoring for system health and performance.

```mermaid
graph TB
    subgraph "Metrics Collection"
        A[Router Statistics] --> E[Metrics Store]
        B[Memory Usage] --> E
        C[Performance Timing] --> E
        D[Error Tracking] --> E
    end
    
    subgraph "Visualization"
        F[Router Heatmaps] --> I[Dashboard]
        G[Memory Plots] --> I
        H[Performance Charts] --> I
    end
    
    subgraph "Alerting"
        J[Memory Threshold] --> M[Alert System]
        K[Performance Degradation] --> M
        L[Error Rate] --> M
    end
    
    E --> F
    E --> G
    E --> H
    E --> J
    E --> K
    E --> L
```

## 🎯 Design Patterns

Key design patterns used throughout the system:

### 1. **Strategy Pattern** - Routers and Adapters
Multiple interchangeable algorithms for routing and adaptation.

### 2. **Factory Pattern** - Component Creation
Centralized creation of adapters, routers, and merge methods.

### 3. **Observer Pattern** - Training Events
Event-driven training with customizable callbacks.

### 4. **Adapter Pattern** - Model Integration
Seamless integration with different model architectures.

### 5. **Composite Pattern** - Layer Composition
Hierarchical composition of fusion layers.

### 6. **Template Method Pattern** - Merge Methods
Consistent interface for different merging algorithms.

## 📚 Implementation Details

### Layer Composition

```python
class MoLLayer(nn.Module):
    """Single MoL fusion layer."""
    
    def __init__(self, experts, adapters, router):
        self.experts = nn.ModuleList(experts)
        self.adapters = nn.ModuleList(adapters)
        self.router = router
    
    def forward(self, x, attention_mask=None):
        # Router decision
        expert_weights, router_logits = self.router(x, attention_mask)
        
        # Expert computation
        expert_outputs = []
        for expert, adapter in zip(self.experts, self.adapters):
            expert_out = expert(x, attention_mask)
            adapted_out = adapter(expert_out)
            expert_outputs.append(adapted_out)
        
        # Weighted combination
        output = sum(w * out for w, out in zip(expert_weights, expert_outputs))
        
        return output, router_logits
```

### Memory Management

```python
class MemoryManager:
    """Efficient memory management for MoL."""
    
    def __init__(self):
        self.model_cache = {}
        self.memory_tracker = MemoryTracker()
    
    def get_model(self, model_name):
        if model_name not in self.model_cache:
            if self.memory_tracker.should_offload():
                self.offload_least_used()
            self.model_cache[model_name] = self.load_model(model_name)
        return self.model_cache[model_name]
```

## 🚀 Performance Characteristics

### Latency Profile

| Component | Latency Impact | Optimization |
|-----------|----------------|--------------|
| Router | Low (1-5ms) | Lightweight MLP |
| Adapter | Low (2-8ms) | Identity initialization |
| Expert Block | High (50-200ms) | Model size dependent |
| Memory Transfer | Medium (10-50ms) | Smart placement |

### Memory Usage

| Configuration | Peak Memory | Optimization |
|---------------|-------------|--------------|
| 2x Small Models | ~2GB | Baseline |
| 2x Medium Models | ~8GB | Offloading |
| 2x Large Models | ~20GB | Distributed |
| 4x Small Models | ~4GB | Lazy loading |

### Throughput Scaling

- **Single GPU**: 10-50 tokens/second (model dependent)
- **Multi-GPU**: Near-linear scaling with model parallelism
- **Distributed**: Scales to 8+ nodes with communication overhead

## 📖 Related Documentation

- [**API Reference**](api/) - Detailed component documentation
- [**Getting Started**](getting-started.md) - Basic usage patterns
- [**Advanced Topics**](advanced/) - Performance optimization
- [**Development Guide**](development.md) - Contributing to architecture

---

**Want to dive deeper?** Explore our [API Reference](api/) or check out [Advanced Topics](advanced/) for optimization techniques!