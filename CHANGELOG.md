# Changelog

All notable changes to Project Mango will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Beta testing phase with community feedback collection
- Enhanced documentation system with professional styling
- GitHub repository setup with proper licensing
- Installation instructions for beta testing from source

## [0.1.0-beta] - 2024-01-15 - **Current Beta Release**

> **🧑‍🔬 Beta Testing**: This is a beta release for testing and feedback. Not yet available on PyPI.

### Added
- Initial release of Project Mango MoL system
- Core MoL runtime for dynamic layer fusion
- Support for multiple model architectures (GPT-2, BERT, T5, etc.)
- Linear and bottleneck adapters for dimension matching
- Simple and token-level routers for expert selection
- Universal architecture handler for automatic model detection
- Model merging capabilities (SLERP, TIES, Task Arithmetic, Linear)
- YAML-based configuration system
- CLI tools for model merging and validation
- SafeTensors integration for secure model serialization
- Hugging Face Hub integration for model sharing
- Memory-efficient training and inference
- Comprehensive test suite
- Example scripts and demonstrations

### Core Features
- **Dynamic Model Fusion**: Runtime combination of transformer layers
- **Universal Architecture Support**: 120+ supported model architectures
- **Memory Optimization**: Lazy loading, offloading, and efficient memory management
- **Security**: Safe model loading with configurable trust levels
- **Scalability**: Support for distributed training and inference
- **Flexibility**: Pluggable adapters and routers for customization

### Supported Models
- **Decoder-Only**: GPT-2, GPT-Neo, GPT-J, LLaMA, Mistral, Falcon, OPT, BLOOM
- **Encoder-Only**: BERT, RoBERTa, DistilBERT, ELECTRA, DeBERTa, ALBERT
- **Encoder-Decoder**: T5, BART, Pegasus, Marian, UL2, FLAN-T5
- **Vision**: ViT, DeiT, Swin Transformer, BEiT, ConvNeXt
- **Multimodal**: CLIP, FLAVA, LayoutLM, LXMERT

### Dependencies
- Python 3.8+
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- Accelerate >= 0.20.0
- SafeTensors >= 0.3.0
- Datasets >= 2.12.0
- Einops >= 0.6.0

## [0.9.0] - 2023-12-01

### Added
- Beta release with core functionality
- Basic model fusion capabilities
- Simple adapter and router implementations
- Initial CLI interface
- Basic documentation

### Known Issues
- Limited architecture support
- Memory optimization needed
- Documentation incomplete

## [0.8.0] - 2023-11-01

### Added
- Alpha release for testing
- Proof of concept implementation
- Basic model loading and fusion
- Experimental features

### Limitations
- Unstable API
- Limited testing
- Development-only release

---

## Version History Summary

| Version | Release Date | Major Features |
|---------|--------------|----------------|
| 1.0.0 | 2024-01-15 | Full release with complete feature set |
| 0.9.0 | 2023-12-01 | Beta with core functionality |
| 0.8.0 | 2023-11-01 | Alpha proof of concept |

## Migration Guides

### Upgrading to 1.0.0 from 0.9.0

#### Breaking Changes
- `MoLConfig` parameter names updated for consistency
- CLI command structure reorganized
- Some utility functions moved to different modules

#### Migration Steps
1. Update configuration files to new parameter names
2. Update import statements for moved utilities
3. Test with new CLI commands
4. Review and update custom components

#### Example Migration
```python
# Old (0.9.0)
from mol.utils import memory_manager
config = MoLConfig(model_list=["gpt2", "bert"])

# New (1.0.0)
from mol.utils.memory_utils import MemoryManager
config = MoLConfig(models=["gpt2", "bert"])
```

### Upgrading to 0.9.0 from 0.8.0

#### Breaking Changes
- Complete API redesign
- New configuration system
- Updated dependencies

#### Migration Steps
1. Review new API documentation
2. Rewrite configuration files
3. Update all import statements
4. Test thoroughly with new API

## Support and Compatibility

### Python Version Support
- **Python 3.8**: Minimum supported version
- **Python 3.9**: Fully supported
- **Python 3.10**: Fully supported
- **Python 3.11**: Fully supported
- **Python 3.12**: Experimental support

### PyTorch Version Support
- **PyTorch 2.0.x**: Minimum supported version
- **PyTorch 2.1.x**: Fully supported
- **PyTorch 2.2.x**: Experimental support

### Platform Support
- **Linux**: Fully supported (Ubuntu 18.04+, CentOS 7+)
- **macOS**: Supported (macOS 10.15+)
- **Windows**: Supported (Windows 10+)
- **Docker**: Official images available

## Contributing

We welcome contributions! Please see our [Development Guide](docs/development.md) for details on:
- Setting up a development environment
- Running tests
- Submitting pull requests
- Code style guidelines

## Security

For security vulnerabilities, please email security@project-mango.dev instead of using the issue tracker.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.