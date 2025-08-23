# Project Mango Documentation 📚

This directory contains the complete documentation for Project Mango's MoL (Modular Layer) system.

## 🏗️ Documentation Structure

```
docs/
├── index.md                     # Main documentation homepage
├── getting-started.md           # Installation and quick start guide
├── configuration.md             # YAML and CLI configuration guide
├── architecture.md              # System architecture overview
├── development.md               # Development and contribution guide
├── tutorials/                   # Step-by-step learning guides
│   ├── index.md                # Tutorials overview
│   ├── basic-fusion.md         # Your first model fusion
│   ├── adapters.md             # Understanding adapters
│   ├── routers.md              # Router basics
│   ├── model-merging.md        # Model merging tutorial
│   ├── training.md             # Training adapters and routers
│   ├── large-models.md         # Working with large models
│   ├── multi-architecture.md   # Multi-architecture fusion
│   ├── custom-components.md    # Building custom components
│   └── deployment.md           # Production deployment
├── examples/                    # Practical code examples
│   ├── index.md                # Examples overview
│   ├── text-generation.md      # Text generation examples
│   ├── question-answering.md   # QA system examples
│   ├── code-generation.md      # Code generation examples
│   └── multi-domain-chat.md    # Multi-domain chatbot
├── api/                         # API reference documentation
│   ├── index.md                # API overview
│   ├── core/                   # Core components API
│   │   ├── mol-runtime.md      # MoLRuntime API
│   │   ├── mol-config.md       # MoLConfig API
│   │   ├── adapters.md         # Adapters API
│   │   ├── routers.md          # Routers API
│   │   ├── block-extractor.md  # Block extraction API
│   │   └── universal-architecture.md # Architecture handler API
│   ├── merge-methods/          # Model merging methods API
│   │   ├── base-merge.md       # Base merge interface
│   │   ├── slerp.md            # SLERP merging
│   │   ├── ties.md             # TIES merging
│   │   ├── task-arithmetic.md  # Task arithmetic
│   │   └── linear.md           # Linear merging
│   ├── training/               # Training system API
│   │   ├── trainer.md          # MoLTrainer API
│   │   └── config.md           # Training configuration
│   ├── utils/                  # Utility functions API
│   │   ├── memory-utils.md     # Memory management
│   │   ├── model-utils.md      # Model utilities
│   │   ├── safetensors-utils.md # SafeTensors integration
│   │   └── hf-utils.md         # Hugging Face utilities
│   └── cli/                    # Command-line interface API
│       ├── merge-cli.md        # Merge CLI documentation
│       ├── validate-cli.md     # Validation CLI
│       └── examples-cli.md     # Examples CLI
├── advanced/                    # Advanced topics and optimization
│   ├── index.md                # Advanced topics overview
│   ├── performance-tuning.md   # Performance optimization
│   ├── memory-management.md    # Advanced memory management
│   ├── distributed-training.md # Multi-GPU training
│   ├── custom-components.md    # Building custom components
│   ├── production-deployment.md # Production deployment
│   ├── security.md             # Security best practices
│   └── research-workflows.md   # Research methodologies
├── stylesheets/                # Custom CSS styles
│   └── extra.css               # Mango theme styles
├── javascripts/                # Custom JavaScript
│   └── extra.js                # Interactive features
└── assets/                     # Images and media
    ├── logo.png                # Project logo
    ├── favicon.png             # Site favicon
    └── diagrams/               # Architecture diagrams
```

## 🎯 Documentation Features

### 📖 Comprehensive Content
- **Getting Started**: From installation to first fusion in minutes
- **Tutorials**: Step-by-step guides for all skill levels
- **Examples**: Practical code samples and use cases
- **API Reference**: Complete function and class documentation
- **Advanced Topics**: Performance optimization and expert techniques

### 🎨 Professional Design
- **Material Design**: Clean, modern interface with Mango branding
- **Mobile Responsive**: Optimized for all screen sizes
- **Dark/Light Mode**: Automatic theme switching
- **Search**: Full-text search across all documentation
- **Navigation**: Intuitive navigation with breadcrumbs and TOC

### 🚀 Interactive Features
- **Code Copying**: One-click code block copying
- **Syntax Highlighting**: Language-specific code highlighting
- **Mermaid Diagrams**: Interactive architecture diagrams
- **Progress Tracking**: Track learning progress through sections
- **Feedback System**: User feedback collection

### 🔧 Technical Features
- **MkDocs**: Built with MkDocs Material theme
- **Git Integration**: Automatic edit links and revision dates
- **SEO Optimized**: Meta tags and social media cards
- **Analytics**: Google Analytics integration
- **Performance**: Optimized loading and caching

## 🛠️ Building Documentation

### Prerequisites
```bash
pip install mkdocs mkdocs-material mkdocstrings[python]
pip install mkdocs-git-revision-date-localized-plugin
pip install mkdocs-minify-plugin
```

### Development Server
```bash
# Serve documentation locally with live reload
mkdocs serve

# Serve on specific port
mkdocs serve --dev-addr=localhost:8001
```

### Building for Production
```bash
# Build static site
mkdocs build

# Build and deploy to GitHub Pages
mkdocs gh-deploy
```

### Configuration
The documentation is configured via `mkdocs.yml` in the project root with:
- **Theme**: Material Design with Mango branding
- **Plugins**: Search, git integration, code documentation
- **Extensions**: Mermaid diagrams, code highlighting, admonitions
- **Navigation**: Structured menu with logical grouping

## 📝 Writing Guidelines

### Documentation Standards
- **Clear Structure**: Use consistent heading hierarchy
- **Code Examples**: Include complete, runnable examples
- **Cross-References**: Link between related sections
- **Visual Aids**: Use diagrams and tables for complex concepts
- **User-Focused**: Write from the user's perspective

### Markdown Conventions
```markdown
# Page Title (H1 - only one per page)

## Major Section (H2)

### Subsection (H3)

#### Detail Section (H4)

- **Bold** for important concepts
- `code` for technical terms
- [Links](./other-page.md) for references

```python
# Complete code examples with comments
from mol import MoLRuntime, MoLConfig

config = MoLConfig(models=["gpt2", "distilgpt2"])
mol = MoLRuntime(config)
```

!!! tip "Pro Tip"
    Use admonitions for tips, warnings, and notes.
```

### Content Types

#### Tutorials
- **Objective**: Clear learning goals
- **Prerequisites**: Required knowledge and setup
- **Step-by-Step**: Numbered instructions with explanations
- **Code Examples**: Complete, tested code samples
- **Troubleshooting**: Common issues and solutions
- **Next Steps**: Links to related content

#### API Documentation
- **Function Signature**: Complete with type hints
- **Parameters**: Description, types, defaults
- **Returns**: Type and description
- **Examples**: Usage examples
- **See Also**: Related functions/classes

#### Examples
- **Complete Code**: Fully working examples
- **Explanation**: What the code does and why
- **Variations**: Different approaches and options
- **Output**: Expected results
- **Extensions**: Ideas for further development

## 🎨 Styling and Branding

### Mango Theme
The documentation uses Project Mango's brand colors:
- **Primary**: Orange (#FF6B35)
- **Secondary**: Golden Orange (#F7931E)
- **Accent**: Yellow (#FFD23F)
- **Dark**: Charcoal (#2E2E2E)
- **Light**: Cream (#FFF8F0)

### Custom Components
- **Cards**: Highlighted content boxes
- **Badges**: Status and category indicators
- **Progress Bars**: Learning progress tracking
- **Feature Grids**: Organized feature displays
- **Difficulty Indicators**: Star-based difficulty levels

## 📊 Analytics and Feedback

### User Tracking
- **Page Views**: Track popular content
- **User Flow**: Understand navigation patterns
- **Search Queries**: Identify content gaps
- **Feedback**: Collect user satisfaction ratings

### Performance Monitoring
- **Load Times**: Monitor page performance
- **Search Performance**: Track search effectiveness
- **Mobile Usage**: Optimize for mobile users
- **Error Tracking**: Identify and fix issues

## 🔄 Maintenance

### Regular Updates
- **Content Review**: Keep information current
- **Link Checking**: Verify all links work
- **Code Testing**: Ensure examples still work
- **Dependency Updates**: Update documentation tools
- **User Feedback**: Address user suggestions

### Version Management
- **Versioned Docs**: Multiple version support
- **Release Notes**: Document changes
- **Migration Guides**: Help users upgrade
- **Deprecation Notices**: Communicate changes

## 🤝 Contributing

### Documentation Contributions
1. **Fork Repository**: Create your own fork
2. **Create Branch**: Use descriptive branch names
3. **Write Content**: Follow our style guidelines
4. **Test Locally**: Verify with `mkdocs serve`
5. **Submit PR**: Create pull request with description

### Content Guidelines
- **Accuracy**: Ensure technical accuracy
- **Clarity**: Write for your audience
- **Completeness**: Cover all necessary details
- **Consistency**: Follow existing patterns
- **Testing**: Verify all code examples work

## 📞 Support

- **GitHub Issues**: Report documentation bugs
- **GitHub Discussions**: Suggest improvements
- **Email**: docs@project-mango.dev
- **Discord**: [Project Mango Community](https://discord.gg/project-mango)

---

**Happy documenting! 📚🥭**