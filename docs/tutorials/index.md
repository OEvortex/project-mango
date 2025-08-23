# Tutorials 📖

Step-by-step guides to master Project Mango's MoL system capabilities.

## 🎯 Learning Path

### 🥇 Beginner Tutorials

Perfect for getting started with MoL fusion concepts:

1. **[Your First Model Fusion](basic-fusion.md)** ⭐  
   Learn the fundamentals by fusing two small models

2. **[Understanding Adapters](adapters.md)**  
   How adapters handle dimension mismatches between models

3. **[Router Basics](routers.md)**  
   Introduction to routing strategies and expert selection

4. **[Configuration Guide](configuration.md)**  
   Master MoL configuration options and best practices

### 🥈 Intermediate Tutorials

Build upon the basics with more advanced scenarios:

5. **[Model Merging with YAML](model-merging.md)**  
   Static model merging using MergeKit-style configurations

6. **[Training Adapters and Routers](training.md)**  
   Fine-tune your fusion for optimal performance  

7. **[Working with Large Models](large-models.md)**  
   Memory optimization and distributed inference techniques

8. **[Multi-Architecture Fusion](multi-architecture.md)**  
   Combine different model architectures (GPT, BERT, T5)

### 🥉 Advanced Tutorials

Master complex scenarios and optimization techniques:

9. **[Custom Components](custom-components.md)**  
   Build your own adapters and routers

10. **[Production Deployment](deployment.md)**  
    Deploy MoL models in production environments

11. **[Performance Optimization](performance.md)**  
    Advanced memory management and speed optimization

12. **[Distributed Training](distributed-training.md)**  
    Scale training across multiple GPUs and nodes

## 🎨 Use Case Tutorials

Real-world applications and domain-specific guides:

### 💬 Natural Language Processing
- **[Building a Multi-Domain Chatbot](nlp/multi-domain-chatbot.md)**
- **[Question Answering with Multiple Models](nlp/question-answering.md)**
- **[Text Generation and Creative Writing](nlp/text-generation.md)**

### 💻 Code and Technical Content
- **[Code Generation with Specialized Models](code/code-generation.md)**
- **[Documentation and Technical Writing](code/technical-writing.md)**
- **[Multi-Language Programming Assistant](code/multi-language.md)**

### 🔬 Research and Experimentation
- **[Model Architecture Comparison](research/architecture-comparison.md)**
- **[Ablation Studies with MoL](research/ablation-studies.md)**
- **[Transfer Learning Experiments](research/transfer-learning.md)**

## 🛠️ Tool-Specific Tutorials

Learn to use MoL's specialized tools and integrations:

### 🖥️ Command Line Interface
- **[CLI Quickstart](tools/cli-quickstart.md)**
- **[Batch Processing with Scripts](tools/batch-processing.md)**
- **[Configuration Management](tools/config-management.md)**

### 🤗 Hugging Face Integration
- **[Uploading Models to Hub](hf/upload-models.md)**
- **[Loading Models from Hub](hf/load-models.md)**
- **[Model Cards and Documentation](hf/model-cards.md)**

### 🔒 SafeTensors Integration
- **[Secure Model Loading](safetensors/secure-loading.md)**
- **[Converting Existing Models](safetensors/conversion.md)**
- **[Performance Benefits](safetensors/performance.md)**

## 🎯 Quick Reference

### Common Patterns

#### Basic Fusion Setup
```python
from mol import MoLRuntime, MoLConfig

config = MoLConfig(models=["gpt2", "distilgpt2"])
mol = MoLRuntime(config)
mol.setup_embeddings()
mol.setup_lm_head()
mol.add_layer([("gpt2", 0), ("distilgpt2", 0)], layer_idx=0)
```

#### Model Merging
```python
from mol.merge_methods import SlerpMerge
from mol.config import MergeConfig

config = MergeConfig(method="slerp", models=["gpt2", "distilgpt2"])
merge = SlerpMerge(config)
merged_model = merge.merge(merge.load_models())
```

#### Training Setup
```python
from mol.training import MoLTrainer, TrainingConfig

training_config = TrainingConfig(learning_rate=1e-4, batch_size=8)
trainer = MoLTrainer(mol, training_config)
trainer.train(train_dataloader)
```

## 📚 Additional Resources

### 📖 Documentation
- [API Reference](../api/) - Complete function documentation
- [Configuration Guide](../configuration.md) - YAML and CLI usage
- [Architecture Overview](../architecture.md) - System design details

### 💡 Examples
- [Code Examples](../examples/) - Working code samples
- [Demo Scripts](../../examples/) - Runnable demonstrations
- [Jupyter Notebooks](../notebooks/) - Interactive tutorials

### 🤝 Community
- [GitHub Discussions](https://github.com/your-username/project-mango/discussions) - Ask questions
- [GitHub Issues](https://github.com/your-username/project-mango/issues) - Report bugs
- [Contributing Guide](../development.md) - Join development

## 🎓 Tutorial Difficulty Levels

| Symbol | Level | Description |
|--------|-------|-------------|
| ⭐ | Beginner | No prior MoL experience needed |
| ⭐⭐ | Intermediate | Basic MoL knowledge required |
| ⭐⭐⭐ | Advanced | Deep understanding of concepts |

## 💡 Learning Tips

1. **Start Sequential**: Follow the numbered beginner tutorials in order
2. **Practice First**: Run the basic examples before advanced topics
3. **Experiment**: Modify examples to understand how parameters affect results
4. **Read API Docs**: Refer to API documentation for detailed parameter explanations
5. **Join Community**: Ask questions in GitHub Discussions

## 🔧 Prerequisites

Before starting the tutorials, ensure you have:

- ✅ Python 3.8+ installed
- ✅ Project Mango installed ([Installation Guide](../getting-started.md))
- ✅ Basic PyTorch knowledge
- ✅ Familiarity with transformer models (recommended)

## 📈 Progress Tracking

Track your learning progress:

- [ ] Complete all Beginner tutorials (1-4)
- [ ] Complete at least 2 Intermediate tutorials (5-8)
- [ ] Choose 1 Advanced tutorial based on your needs (9-12)
- [ ] Try 1 Use Case tutorial relevant to your domain
- [ ] Explore Tool-Specific tutorials as needed

---

**Ready to start learning?** Begin with [Your First Model Fusion](basic-fusion.md)! 🚀