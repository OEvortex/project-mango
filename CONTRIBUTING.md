# Contributing to Project Mango 🥭

Thank you for your interest in contributing to Project Mango! This document provides guidelines and information for contributors.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Community](#community)

## 🤝 Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to conduct@project-mango.dev.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- PyTorch 2.0+ (automatically installed)
- Basic knowledge of transformer models and PyTorch

### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/OEvortex/project-mango.git
   cd project-mango
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```

5. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

6. **Verify your setup**:
   ```bash
   python -m pytest tests/ -v
   python examples/basic_fusion_demo.py
   ```

## 🛠️ How to Contribute

### Types of Contributions

We welcome several types of contributions:

#### 🐛 Bug Reports
- Use the bug report template
- Include minimal reproduction steps
- Provide system information
- Check if the issue already exists

#### ✨ Feature Requests
- Use the feature request template
- Explain the use case and benefits
- Consider implementation complexity
- Discuss with maintainers first for large features

#### 🔧 Code Contributions
- Bug fixes
- New features
- Performance improvements
- Code refactoring

#### 📚 Documentation
- Fix typos and errors
- Improve existing documentation
- Add new tutorials or examples
- Translate documentation

#### 🧪 Testing
- Add missing tests
- Improve test coverage
- Performance benchmarks
- Integration tests

### What We're Looking For

#### High Priority
- 🐛 Bug fixes for reported issues
- 📖 Documentation improvements
- 🧪 Test coverage improvements
- 🚀 Performance optimizations
- 🔒 Security enhancements

#### Medium Priority
- ✨ New adapter implementations
- 🧭 New router strategies
- 🔀 New merge methods
- 🏗️ Architecture support extensions
- 💡 Example applications

#### Future Considerations
- 🌐 Web interface development
- 📱 Mobile deployment support
- ☁️ Cloud integration features
- 🔧 Developer tools and utilities

## 🔄 Development Workflow

### 1. Create an Issue

Before starting work:
- Check existing issues and PRs
- Create an issue describing your contribution
- Discuss approach with maintainers
- Get approval for significant changes

### 2. Create a Branch

```bash
# Create and switch to a new branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-number-description
```

### 3. Make Changes

- Follow our coding standards
- Write tests for new functionality
- Update documentation as needed
- Keep commits focused and atomic

### 4. Test Your Changes

```bash
# Run the full test suite
python -m pytest

# Run specific tests
python -m pytest tests/test_adapters.py -v

# Check code coverage
python -m pytest --cov=mol --cov-report=html

# Run linting
black --check mol/ tests/
isort --check-only mol/ tests/
flake8 mol/ tests/
mypy mol/

# Test examples
python examples/basic_fusion_demo.py
python examples/comprehensive_demo.py --small-models
```

### 5. Commit Your Changes

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Format: type(scope): description
git commit -m "feat(adapters): add attention-based adapter implementation"
git commit -m "fix(routers): resolve numerical instability in softmax"
git commit -m "docs(tutorials): add advanced training tutorial"
git commit -m "test(core): add tests for memory management"
```

#### Commit Types
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `style`: Code style/formatting
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements

### 6. Push and Create PR

```bash
# Push your branch
git push origin feature/your-feature-name

# Create pull request via GitHub UI
```

## 📝 Coding Standards

### Python Style

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 88 characters (Black default)
- **Imports**: Use `isort` for import sorting
- **Type hints**: Required for all public functions
- **Docstrings**: Google style docstrings

### Code Formatting

We use automated tools for consistency:

```bash
# Format code
black mol/ tests/ examples/
isort mol/ tests/ examples/

# Check formatting
black --check mol/
isort --check-only mol/
flake8 mol/
mypy mol/
```

### Example Code Style

```python
from typing import Optional, List, Tuple
import torch
import torch.nn as nn


class ExampleAdapter(BaseAdapter):
    """
    Example adapter implementation.
    
    This adapter demonstrates proper code style and documentation
    conventions for Project Mango.
    
    Args:
        input_dim: Input dimension size
        output_dim: Output dimension size
        dropout_rate: Dropout probability for regularization
        
    Attributes:
        projection: Linear projection layer
        dropout: Dropout layer for regularization
        
    Example:
        >>> adapter = ExampleAdapter(512, 768, dropout_rate=0.1)
        >>> output = adapter(input_tensor)
    """
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        dropout_rate: float = 0.1
    ) -> None:
        super().__init__(input_dim, output_dim)
        
        # Initialize components
        self.projection = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize adapter weights for stable training."""
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the adapter.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, output_dim]
            
        Raises:
            ValueError: If input tensor has incorrect dimensions
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input tensor, got {x.dim()}D")
        
        # Apply transformation
        output = self.projection(x)
        output = self.dropout(output)
        
        return output
```

### Error Handling

```python
# Good: Specific error handling with informative messages
try:
    model = load_model(model_name)
except FileNotFoundError:
    raise FileNotFoundError(f"Model '{model_name}' not found. Please check the path.")
except Exception as e:
    logger.error(f"Failed to load model '{model_name}': {e}")
    raise RuntimeError(f"Model loading failed: {e}") from e

# Good: Input validation
def create_adapter(adapter_type: str, **kwargs) -> BaseAdapter:
    """Create adapter with validation."""
    if not isinstance(adapter_type, str):
        raise TypeError(f"adapter_type must be string, got {type(adapter_type)}")
    
    if adapter_type not in SUPPORTED_ADAPTERS:
        raise ValueError(
            f"Unsupported adapter type '{adapter_type}'. "
            f"Supported types: {list(SUPPORTED_ADAPTERS.keys())}"
        )
```

## 🧪 Testing Guidelines

### Test Structure

```python
import pytest
import torch
from mol.core.adapters import LinearAdapter


class TestLinearAdapter:
    """Test suite for LinearAdapter class."""
    
    @pytest.fixture
    def adapter(self):
        """Create adapter for testing."""
        return LinearAdapter(input_dim=512, output_dim=768)
    
    def test_initialization(self, adapter):
        """Test proper initialization."""
        assert adapter.input_dim == 512
        assert adapter.output_dim == 768
        assert isinstance(adapter.projection, nn.Linear)
    
    def test_forward_pass(self, adapter):
        """Test forward pass with valid input."""
        x = torch.randn(2, 10, 512)
        output = adapter(x)
        
        assert output.shape == (2, 10, 768)
        assert not torch.isnan(output).any()
    
    def test_invalid_input(self, adapter):
        """Test error handling for invalid input."""
        x = torch.randn(512)  # Wrong dimensions
        
        with pytest.raises(ValueError, match="Expected 3D input tensor"):
            adapter(x)
    
    @pytest.mark.gpu
    def test_cuda_compatibility(self, adapter):
        """Test CUDA device compatibility."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        adapter = adapter.cuda()
        x = torch.randn(2, 10, 512).cuda()
        output = adapter(x)
        
        assert output.is_cuda
        assert output.device == x.device
    
    @pytest.mark.slow
    def test_performance_benchmark(self, adapter):
        """Benchmark adapter performance."""
        x = torch.randn(32, 100, 512)
        
        # Warmup
        for _ in range(10):
            adapter(x)
        
        # Benchmark
        import time
        start_time = time.time()
        for _ in range(100):
            adapter(x)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        assert avg_time < 0.01  # Should be fast
```

### Test Categories

Use pytest markers to categorize tests:

```python
# Unit tests (default)
def test_basic_functionality():
    pass

# GPU tests
@pytest.mark.gpu
def test_cuda_operations():
    pass

# Slow tests (integration, benchmarks)
@pytest.mark.slow
def test_full_training_pipeline():
    pass

# Network tests (require internet)
@pytest.mark.network
def test_model_download():
    pass
```

### Running Tests

```bash
# All tests
pytest

# Specific categories
pytest -m "not slow"  # Skip slow tests
pytest -m gpu        # Only GPU tests
pytest -k adapter    # Tests matching "adapter"

# With coverage
pytest --cov=mol --cov-report=html
```

## 📚 Documentation

### Docstring Style

We use Google-style docstrings:

```python
def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    epochs: int,
    learning_rate: float = 1e-4
) -> Dict[str, float]:
    """
    Train a PyTorch model with the given configuration.
    
    This function provides a complete training loop with progress tracking,
    loss computation, and metric collection.
    
    Args:
        model: PyTorch model to train
        dataloader: DataLoader providing training data
        epochs: Number of training epochs
        learning_rate: Learning rate for optimization (default: 1e-4)
        
    Returns:
        Dictionary containing training metrics:
            - 'final_loss': Final training loss
            - 'avg_loss': Average loss across all epochs
            - 'total_time': Total training time in seconds
            
    Raises:
        ValueError: If epochs <= 0 or learning_rate <= 0
        RuntimeError: If model training fails
        
    Example:
        >>> model = MyModel()
        >>> dataloader = create_dataloader(data)
        >>> metrics = train_model(model, dataloader, epochs=10)
        >>> print(f"Final loss: {metrics['final_loss']:.4f}")
        
    Note:
        This function modifies the model in-place. Consider creating
        a copy if you need to preserve the original state.
    """
```

### Documentation Updates

When contributing:

1. **Update docstrings** for any new or modified functions
2. **Add examples** for new features in `examples/`
3. **Update tutorials** if the change affects user workflows
4. **Update API docs** if public interfaces change
5. **Add changelog entries** for user-visible changes

### Documentation Building

```bash
# Install documentation tools
pip install mkdocs mkdocs-material mkdocstrings[python]

# Serve documentation locally
mkdocs serve

# Build documentation
mkdocs build
```

## 🏗️ Adding New Components

### New Adapter

1. Create the adapter class inheriting from `BaseAdapter`
2. Implement required methods (`__init__`, `forward`)
3. Add to the adapter factory function
4. Write comprehensive tests
5. Add documentation and examples
6. Update configuration options

### New Router

1. Create the router class inheriting from `BaseRouter`
2. Implement required methods (`__init__`, `forward`)
3. Add to the router factory function
4. Write comprehensive tests
5. Add documentation and examples
6. Update configuration options

### New Merge Method

1. Create the merge class inheriting from `BaseMerge`
2. Implement required methods (`merge`, `validate_config`)
3. Add to the merge method factory
4. Write comprehensive tests
5. Add CLI support
6. Update documentation

## 🚀 Performance Guidelines

### Memory Efficiency

```python
# Good: Use context managers for memory cleanup
@contextmanager
def temporary_model_loading():
    model = load_large_model()
    try:
        yield model
    finally:
        del model
        torch.cuda.empty_cache()

# Good: Implement lazy loading
class LazyModelLoader:
    def __init__(self, model_name):
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self):
        if self._model is None:
            self._model = load_model(self.model_name)
        return self._model
```

### Speed Optimization

```python
# Good: Use torch.jit.script for performance-critical code
@torch.jit.script
def optimized_attention(query, key, value):
    scores = torch.matmul(query, key.transpose(-2, -1))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, value)

# Good: Profile performance-critical paths
def profile_function(func):
    import time
    start = time.time()
    result = func()
    end = time.time()
    print(f"{func.__name__} took {end - start:.4f} seconds")
    return result
```

## 🔒 Security Guidelines

### Input Validation

```python
# Always validate inputs
def safe_model_loading(model_path: str, trust_remote_code: bool = False):
    if not isinstance(model_path, str):
        raise TypeError("model_path must be a string")
    
    if not trust_remote_code and contains_remote_code(model_path):
        raise SecurityError(
            "Model contains remote code but trust_remote_code=False. "
            "Set trust_remote_code=True only for trusted models."
        )
```

### Safe Defaults

```python
# Use safe defaults
@dataclass
class MoLConfig:
    models: List[str]
    trust_remote_code: bool = False  # Secure by default
    validate_models: bool = True     # Validate by default
    max_memory_gb: Optional[int] = None  # No unlimited memory
```

## 🐛 Debugging Tips

### Common Issues

1. **CUDA out of memory**:
   ```python
   # Enable memory efficient mode
   config.memory_efficient = True
   config.use_gradient_checkpointing = True
   ```

2. **Model loading errors**:
   ```python
   # Add verbose logging
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

3. **Numerical instabilities**:
   ```python
   # Check for NaN/inf values
   if torch.isnan(tensor).any():
       logger.warning("NaN detected in tensor")
   ```

## 📞 Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Discord**: Real-time community chat
- **Email**: maintainers@project-mango.dev

### Before Asking for Help

1. Search existing issues and discussions
2. Check the documentation
3. Try the troubleshooting guides
4. Provide minimal reproduction examples
5. Include system information and error messages

## 🎉 Recognition

Contributors are recognized through:

- **Contributors file**: Listed in CONTRIBUTORS.md
- **Release notes**: Mentioned in changelog
- **GitHub**: Automatic contribution tracking
- **Documentation**: Attribution for significant contributions

## 📋 Checklist for Pull Requests

Before submitting a PR, ensure:

- [ ] Code follows our style guidelines
- [ ] Tests are added for new functionality
- [ ] All tests pass locally
- [ ] Documentation is updated
- [ ] Commit messages follow conventions
- [ ] PR description is clear and complete
- [ ] Breaking changes are documented
- [ ] Performance impact is considered

## 📄 License

By contributing to Project Mango, you agree that your contributions will be licensed under the Apache License 2.0.

---

Thank you for contributing to Project Mango! Your efforts help make this project better for everyone. 🥭