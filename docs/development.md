# Development Guide 👥

Comprehensive guide for contributing to Project Mango's MoL system development.

## 🎯 Welcome Contributors!

Thank you for your interest in contributing to Project Mango! This guide will help you set up a development environment, understand our codebase, and make meaningful contributions.

## 🚀 Quick Start for Developers

### Prerequisites

- **Python 3.8+** with pip
- **Git** for version control
- **PyTorch 2.0+** (will be installed automatically)
- **CUDA-compatible GPU** (optional, but recommended)

### Development Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/your-username/project-mango.git
cd project-mango

# 2. Create development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install in development mode
pip install -e ".[dev]"

# 4. Install pre-commit hooks
pre-commit install

# 5. Verify installation
python -m pytest tests/ -v
python examples/basic_fusion_demo.py
```

### Development Dependencies

```bash
# Core dependencies (automatically installed)
pip install torch>=2.0.0 transformers>=4.30.0 datasets accelerate

# Development tools
pip install pytest pytest-cov black isort flake8 mypy pre-commit

# Documentation tools  
pip install mkdocs mkdocs-material mkdocstrings

# Optional tools
pip install wandb jupyterlab ipython
```

## 📁 Project Structure

Understanding the codebase organization:

```
project-mango/
├── mol/                           # Main package
│   ├── __init__.py               # Package initialization
│   ├── core/                     # Core MoL components
│   │   ├── __init__.py
│   │   ├── mol_runtime.py        # Main runtime orchestrator
│   │   ├── adapters.py           # Dimension matching adapters
│   │   ├── routers.py            # Expert selection routers
│   │   ├── block_extractor.py    # Model layer extraction
│   │   └── universal_architecture.py  # Architecture handling
│   ├── merge_methods/            # Model merging algorithms
│   │   ├── __init__.py
│   │   ├── base_merge.py         # Base merging interface
│   │   ├── slerp.py              # SLERP implementation
│   │   ├── ties.py               # TIES implementation
│   │   ├── task_arithmetic.py    # Task arithmetic
│   │   └── linear.py             # Linear merging
│   ├── config/                   # Configuration system
│   │   ├── __init__.py
│   │   ├── merge_config.py       # Merge configurations
│   │   ├── config_parser.py      # YAML parsing
│   │   └── validation.py         # Config validation
│   ├── training/                 # Training pipeline
│   │   ├── __init__.py
│   │   └── trainer.py            # Main trainer class
│   ├── utils/                    # Utility functions
│   │   ├── __init__.py
│   │   ├── memory_utils.py       # Memory management
│   │   ├── model_utils.py        # Model utilities
│   │   ├── safetensors_utils.py  # SafeTensors integration
│   │   └── hf_utils.py           # Hugging Face utilities
│   ├── cli/                      # Command-line interface
│   │   ├── __init__.py
│   │   ├── merge_cli.py          # Main CLI
│   │   ├── validate_cli.py       # Config validation
│   │   └── examples_cli.py       # Example generation
│   └── models/                   # Model loading and management
│       ├── __init__.py
│       ├── base_model.py         # Base model interface
│       └── huggingface_models.py # HF integration
├── examples/                     # Example scripts
├── tests/                        # Test suite
├── docs/                         # Documentation
├── requirements.txt              # Dependencies
├── setup.py                      # Package setup
├── pyproject.toml               # Project configuration
└── README.md                    # Project overview
```

## 🧪 Testing Framework

We use pytest for comprehensive testing with multiple test categories.

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=mol --cov-report=html

# Run specific test categories
python -m pytest tests/test_adapters.py -v
python -m pytest tests/test_routers.py -v
python -m pytest tests/test_merge_methods.py -v

# Run integration tests
python -m pytest tests/integration/ -v

# Run with specific markers
python -m pytest -m "not slow"  # Skip slow tests
python -m pytest -m "gpu"       # Only GPU tests
```

### Test Structure

```python
# tests/test_adapters.py
import pytest
import torch
from mol.core.adapters import LinearAdapter, BottleneckAdapter

class TestLinearAdapter:
    """Test suite for LinearAdapter."""
    
    def test_same_dimension_identity(self):
        """Test identity behavior for same dimensions."""
        adapter = LinearAdapter(768, 768, init_identity=True)
        x = torch.randn(2, 10, 768)
        output = adapter(x)
        
        # Should be close to identity
        assert torch.allclose(output, x, atol=1e-2)
    
    def test_dimension_projection(self):
        """Test dimension projection."""
        adapter = LinearAdapter(512, 768)
        x = torch.randn(2, 10, 512)
        output = adapter(x)
        
        assert output.shape == (2, 10, 768)
    
    @pytest.mark.gpu
    def test_cuda_compatibility(self):
        """Test CUDA device compatibility."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        adapter = LinearAdapter(768, 768).cuda()
        x = torch.randn(2, 10, 768).cuda()
        output = adapter(x)
        
        assert output.is_cuda
        assert output.device == x.device
```

### Adding New Tests

When adding new functionality, include:

1. **Unit Tests**: Test individual functions/classes
2. **Integration Tests**: Test component interactions  
3. **Performance Tests**: Benchmark critical paths
4. **GPU Tests**: Verify CUDA compatibility
5. **Error Tests**: Test error handling and edge cases

```python
# Example test template
def test_new_feature():
    """Test description."""
    # Arrange
    setup_data = create_test_data()
    
    # Act  
    result = new_feature(setup_data)
    
    # Assert
    assert result.meets_expectations()
    assert no_side_effects_occurred()
```

## 🎨 Code Style and Standards

We follow strict coding standards to maintain code quality and consistency.

### Code Formatting

```bash
# Auto-format code
black mol/ tests/ examples/
isort mol/ tests/ examples/

# Check formatting
black --check mol/
isort --check-only mol/

# Lint code
flake8 mol/ tests/
mypy mol/
```

### Style Guidelines

#### Python Code Style

```python
# Good: Clear function with type hints and docstring
def create_adapter(
    adapter_type: str,
    input_dim: int,
    output_dim: int,
    init_identity: bool = True
) -> BaseAdapter:
    """
    Create an adapter instance.
    
    Args:
        adapter_type: Type of adapter ('linear' or 'bottleneck')
        input_dim: Input dimension
        output_dim: Output dimension
        init_identity: Whether to use identity initialization
        
    Returns:
        Configured adapter instance
        
    Raises:
        ValueError: If adapter_type is unsupported
    """
    if adapter_type == "linear":
        return LinearAdapter(input_dim, output_dim, init_identity=init_identity)
    elif adapter_type == "bottleneck":
        return BottleneckAdapter(input_dim, output_dim, init_identity=init_identity)
    else:
        raise ValueError(f"Unsupported adapter type: {adapter_type}")
```

#### Documentation Standards

```python
class MoLRuntime(nn.Module):
    """
    Main MoL Runtime for dynamic layer fusion.
    
    This class orchestrates the fusion of transformer layers from different
    models using adapters and routing mechanisms.
    
    Args:
        config: MoL configuration specifying models and fusion parameters
        
    Attributes:
        config: The configuration object
        layers: List of MoL fusion layers
        tokenizer: Tokenizer from the primary model
        
    Example:
        >>> config = MoLConfig(models=["gpt2", "distilgpt2"])
        >>> mol = MoLRuntime(config)
        >>> mol.setup_embeddings()
        >>> mol.setup_lm_head()
    """
```

### Pre-commit Hooks

We use pre-commit hooks to enforce code quality:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.8

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203,W503]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [torch, transformers]
```

## 🔧 Development Workflow

### Git Workflow

We use GitHub Flow for development:

```bash
# 1. Create feature branch
git checkout -b feature/new-adapter-type

# 2. Make changes and commit
git add .
git commit -m "feat: Add attention-based adapter

- Implement AttentionAdapter class
- Add tests for attention mechanism
- Update documentation"

# 3. Push and create PR
git push origin feature/new-adapter-type
# Create PR via GitHub UI

# 4. Address review feedback
git add .
git commit -m "fix: Address review comments"
git push origin feature/new-adapter-type

# 5. Merge after approval
# PR will be merged via GitHub
```

### Commit Message Format

```
type(scope): Short description

Longer description if needed:
- Bullet point 1
- Bullet point 2

Closes #123
```

**Types:**
- `feat`: New features
- `fix`: Bug fixes  
- `docs`: Documentation changes
- `style`: Code style/formatting
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

### Pull Request Process

1. **Create Issue**: Describe the problem or feature request
2. **Fork Repository**: Create your own fork  
3. **Create Branch**: Use descriptive branch names
4. **Implement Changes**: Follow coding standards
5. **Add Tests**: Ensure adequate test coverage
6. **Update Documentation**: Keep docs current
7. **Create PR**: Use our PR template
8. **Address Reviews**: Respond to feedback promptly
9. **Merge**: Squash merge after approval

### PR Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that breaks existing functionality)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Performance impact assessed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No breaking changes (or documented)
```

## 🏗️ Adding New Components

### Creating a New Adapter

```python
# mol/core/adapters.py
class MyCustomAdapter(BaseAdapter):
    """Custom adapter implementation."""
    
    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        super().__init__(input_dim, output_dim)
        # Initialize your adapter
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass implementation."""
        # Your implementation here
        return x

# Register the adapter
def create_adapter(adapter_type: str, **kwargs) -> BaseAdapter:
    if adapter_type == "my_custom":
        return MyCustomAdapter(**kwargs)
    # ... existing types
```

### Creating a New Router

```python
# mol/core/routers.py
class MyCustomRouter(BaseRouter):
    """Custom router implementation."""
    
    def __init__(self, hidden_dim: int, num_experts: int, **kwargs):
        super().__init__(hidden_dim, num_experts)
        # Initialize your router
        
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass implementation."""
        # Your implementation here
        return expert_weights, router_logits

# Register the router
def create_router(router_type: str, **kwargs) -> BaseRouter:
    if router_type == "my_custom":
        return MyCustomRouter(**kwargs)
    # ... existing types
```

### Creating a New Merge Method

```python
# mol/merge_methods/my_method.py
from .base_merge import BaseMerge

class MyMergeMethod(BaseMerge):
    """Custom merge method implementation."""
    
    def merge(self, models: List[torch.nn.Module]) -> torch.nn.Module:
        """Implement your merging logic."""
        # Your implementation here
        pass

# Register in __init__.py
def create_merge_method(config: MergeConfig) -> BaseMerge:
    if config.method == "my_method":
        return MyMergeMethod(config)
    # ... existing methods
```

## 📊 Performance Guidelines

### Memory Optimization

```python
# Good: Memory-efficient implementation
class EfficientAdapter(BaseAdapter):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use in-place operations when possible
        x = F.gelu(x, inplace=True)
        
        # Clear intermediate variables
        intermediate = self.projection(x)
        del x  # Explicit cleanup for large tensors
        
        return intermediate

# Good: Context managers for memory
@contextmanager
def memory_efficient_forward():
    """Context manager for memory-efficient operations."""
    try:
        torch.cuda.empty_cache()
        yield
    finally:
        torch.cuda.empty_cache()
```

### Performance Profiling

```python
# Use profiling for optimization
def profile_component():
    """Profile component performance."""
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        # Your code here
        pass
    
    # Analyze results
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

## 🧪 Debugging and Development Tools

### Debugging Tips

```python
# Use logging for debugging
import logging
logger = logging.getLogger(__name__)

def debug_router_weights(expert_weights: torch.Tensor):
    """Debug router weight distribution."""
    logger.debug(f"Expert weights shape: {expert_weights.shape}")
    logger.debug(f"Weight distribution: {expert_weights.mean(dim=0)}")
    logger.debug(f"Weight std: {expert_weights.std(dim=0)}")
    
    # Check for degenerate cases
    if (expert_weights < 1e-6).all(dim=-1).any():
        logger.warning("Detected near-zero router weights")
```

### Development Scripts

```bash
# scripts/dev_setup.sh - Development environment setup
#!/bin/bash
set -e

echo "Setting up development environment..."

# Install dependencies
pip install -e ".[dev]"

# Setup pre-commit
pre-commit install

# Run tests
python -m pytest tests/ -v

# Generate documentation
mkdocs build

echo "Development environment ready!"
```

### Useful Development Commands

```bash
# Quick development commands
make test          # Run tests
make lint          # Run linting
make format        # Format code
make docs          # Build documentation
make clean         # Clean build artifacts

# Performance testing
python scripts/benchmark.py

# Memory profiling
python scripts/memory_profile.py

# Generate coverage report
python -m pytest --cov=mol --cov-report=html
open htmlcov/index.html
```

## 📚 Documentation Contributions

### Adding Documentation

```bash
# Documentation structure
docs/
├── index.md                 # Main index
├── getting-started.md       # Getting started guide
├── tutorials/              # Step-by-step tutorials
├── examples/               # Code examples
├── api/                    # API reference
└── advanced/               # Advanced topics
```

### Documentation Standards

```markdown
# Page Title

Brief description of the page content.

## Section Header

Detailed explanation with:

- **Bold important concepts**
- `code snippets` for technical terms
- Links to [related pages](./other-page.md)

### Code Examples

```python
# Always include complete, runnable examples
from mol import MoLRuntime, MoLConfig

config = MoLConfig(models=["gpt2", "distilgpt2"])
mol = MoLRuntime(config)
```

### Tips and Warnings

> **💡 Tip**: Use callouts for important information

> **⚠️ Warning**: Highlight potential issues

> **📝 Note**: Additional context or clarification
```

## 🚀 Release Process

### Version Management

We use semantic versioning (SemVer):

- **Major** (1.0.0): Breaking changes
- **Minor** (0.1.0): New features (backward compatible)
- **Patch** (0.0.1): Bug fixes

### Release Checklist

1. **Update Version**: Bump version in `setup.py` and `__init__.py`
2. **Update Changelog**: Document all changes since last release
3. **Run Full Test Suite**: Ensure all tests pass
4. **Update Documentation**: Ensure docs are current
5. **Create Release PR**: Review all changes
6. **Tag Release**: Create Git tag after merge
7. **Build and Upload**: Build package and upload to PyPI
8. **Update Documentation**: Deploy updated docs

## 🤝 Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment:

- **Be respectful**: Treat all contributors with respect
- **Be inclusive**: Welcome people of all backgrounds
- **Be collaborative**: Work together constructively
- **Be patient**: Help newcomers learn and grow
- **Be professional**: Maintain professional standards

### Getting Help

- **Documentation**: Check our comprehensive docs first
- **GitHub Discussions**: Ask questions and share ideas
- **GitHub Issues**: Report bugs and request features
- **Code Reviews**: Learn from feedback and provide constructive reviews

### Recognition

We appreciate all contributors! Contributors are recognized in:

- **CONTRIBUTORS.md**: List of all contributors
- **Release Notes**: Recognition for significant contributions
- **GitHub**: Automatic contribution tracking
- **Documentation**: Attribution for major doc contributions

## 📞 Contact and Support

- **GitHub Issues**: Technical problems and feature requests
- **GitHub Discussions**: Questions and community discussions
- **Email**: maintainers@project-mango.dev
- **Discord**: [Project Mango Community](https://discord.gg/project-mango)

---

**Ready to contribute?** Check out our [Good First Issues](https://github.com/your-username/project-mango/labels/good%20first%20issue) or ask questions in [Discussions](https://github.com/your-username/project-mango/discussions)!

**Thank you for helping make Project Mango better! 🥭**