#!/usr/bin/env python3
"""
Project Mango - Qwen Medical Expert Fusion Training Script

This script demonstrates advanced MoL fusion of:
- Qwen/Qwen3-0.6B (General purpose model)
- suayptalha/Qwen3-0.6B-Medical-Expert (Medical specialist model)

Features:
- 10-layer fusion with intelligent routing
- Comprehensive training pipeline
- Memory-efficient implementation
- Hugging Face Hub integration
- SafeTensors support
- Detailed logging and monitoring
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
import argparse
from pathlib import Path
import json
import time
from typing import List, Dict, Any, Optional
import os
from datetime import datetime

# MoL imports
from mol import MoLRuntime
from mol.core.mol_runtime import MoLConfig
from mol.training.trainer import MoLTrainer, TrainingConfig, create_simple_dataset
from mol.utils.hf_utils import HuggingFacePublisher
from mol.utils.model_utils import ModelUtils
from mol.utils.memory_utils import MemoryManager

# External imports
from transformers import AutoTokenizer
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

try:
    from datasets import Dataset, load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    Dataset = None

# Setup logging with UTF-8 encoding and Windows console support
import sys

# Create console handler with proper encoding
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Create file handler with UTF-8 encoding
file_handler = logging.FileHandler('qwen_fusion_training.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)

# Set encoding for stdout if possible (Windows)
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except:
    pass  # Ignore encoding setup errors
logger = logging.getLogger(__name__)


def safe_log(level, message, *args, **kwargs):
    """Safely log messages by removing problematic Unicode characters."""
    try:
        # Convert message to string and encode/decode to remove problematic characters
        if isinstance(message, str):
            # Remove or replace problematic Unicode characters
            safe_message = message.encode('ascii', errors='replace').decode('ascii')
        else:
            safe_message = str(message)
        
        # Call the appropriate logging method
        getattr(logger, level)(safe_message, *args, **kwargs)
    except Exception:
        # Fallback to basic logging
        getattr(logger, level)(str(message).encode('ascii', errors='replace').decode('ascii'))


class QwenMedicalFusionTrainer:
    """
    Advanced trainer for Qwen medical expert fusion.
    """
    
    def __init__(
        self,
        output_dir: str = "./qwen_fusion_output",
        use_wandb: bool = True,
        push_to_hub: bool = True,
        hub_repo_id: Optional[str] = None,
        hub_private: bool = False
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_wandb = use_wandb
        self.push_to_hub = push_to_hub
        self.hub_repo_id = hub_repo_id
        self.hub_private = hub_private
        
        # Training metrics
        self.training_metrics = {}
        self.start_time = None
        
        # Initialize components
        self.mol_runtime = None
        self.trainer = None
        self.hf_publisher = None
        
        logger.info("Qwen Medical Fusion Trainer initialized")
    
    def create_mol_config(self) -> MoLConfig:
        """Create optimized MoL configuration for Qwen fusion."""
        
        config = MoLConfig(
            models=[
                'Qwen/Qwen3-0.6B',                    # General purpose expert
                'suayptalha/Qwen3-0.6B-Medical-Expert'  # Medical domain expert
            ],
            adapter_type="bottleneck",  # More parameter efficient
            router_type="token_level",  # Per-token expert selection for fine-grained routing
            max_layers=10,              # Use 10 layers as requested
            target_hidden_dim=None,     # Auto-detect from largest model
            use_gradient_checkpointing=True,  # Memory efficiency
            memory_efficient=True,
            temperature=1.0,
            entropy_penalty_coeff=0.02,  # Encourage diverse routing
            load_balancing_coeff=0.01,   # Balance expert usage
            top_k_experts=None,          # Use both experts (no top-k)
            device_map="auto",           # Automatic device placement
            trust_remote_code=False      # Security first
        )
        
        logger.info(f"MoL Config - Models: {config.models}")
        logger.info(f"MoL Config - Layers: {config.max_layers}, Adapter: {config.adapter_type}, Router: {config.router_type}")
        
        return config
    
    def create_training_config(self, epochs: int = 5, batch_size: int = 4) -> TrainingConfig:
        """Create optimized training configuration."""
        
        config = TrainingConfig(
            # Learning rates - separate rates for different components
            learning_rate=2e-5,          # Main adapter learning rate
            router_learning_rate=1e-4,   # Higher LR for routers (they need to learn routing quickly)
            weight_decay=0.01,
            
            # Training schedule
            batch_size=batch_size,
            max_epochs=epochs,
            warmup_steps=100,
            
            # Logging and checkpointing
            logging_steps=25,
            eval_steps=200,
            save_steps=500,
            
            # Optimization
            gradient_clip_norm=1.0,
            entropy_penalty_coeff=0.02,  # Match MoL config
            load_balancing_coeff=0.01,   # Match MoL config
            
            # Training strategy
            freeze_experts=True,         # Keep original models frozen
            use_gradient_checkpointing=True,  # Memory efficiency
            use_safetensors=True,        # Secure serialization
            
            # Output and monitoring
            output_dir=str(self.output_dir / "checkpoints"),
            run_name=f"qwen_medical_fusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            use_wandb=self.use_wandb and WANDB_AVAILABLE
        )
        
        logger.info(f"Training Config - Epochs: {config.max_epochs}, Batch Size: {config.batch_size}")
        logger.info(f"Training Config - LR: {config.learning_rate}, Router LR: {config.router_learning_rate}")
        
        return config
    
    def create_training_dataset(self, num_samples: int = 2000) -> tuple[list, list]:
        """Create comprehensive training dataset for medical + general fusion."""
        
        logger.info("Creating training dataset...")
        
        # Medical domain texts (specialized knowledge)
        medical_texts = [
            "The patient presents with acute myocardial infarction requiring immediate intervention.",
            "Differential diagnosis includes pneumonia, pulmonary embolism, and pleuritis.",
            "The pharmacokinetics of this medication show hepatic metabolism via CYP450.",
            "Clinical symptoms suggest autoimmune etiology requiring immunosuppressive therapy.",
            "The radiological findings are consistent with acute appendicitis diagnosis.",
            "Laboratory results indicate elevated inflammatory markers and leukocytosis.",
            "The surgical procedure requires general anesthesia and cardiac monitoring.",
            "Patient history reveals familial predisposition to cardiovascular disease.",
            "The therapeutic regimen includes beta-blockers and ACE inhibitors.",
            "Diagnostic imaging shows contrast enhancement suggesting malignancy.",
        ]
        
        # General domain texts (broad knowledge)
        general_texts = [
            "Hello, how can I help you today with your questions?",
            "The weather forecast predicts sunny skies and mild temperatures.",
            "Technology continues to advance rapidly in various fields.",
            "Education plays a crucial role in personal development.",
            "Travel broadens perspectives and creates lasting memories.",
            "Cooking is both an art form and a practical skill.",
            "Music has the power to evoke strong emotions.",
            "Reading books expands vocabulary and knowledge.",
            "Exercise contributes significantly to overall health.",
            "Communication skills are essential in professional settings.",
        ]
        
        # Mixed domain texts (require intelligent routing)
        mixed_texts = [
            "Can you explain the medical condition and its treatment options?",
            "What are the symptoms of diabetes and how is it managed?",
            "How does the cardiovascular system function in healthy individuals?",
            "Please describe the process of cellular respiration in detail.",
            "What safety precautions should be taken during surgery?",
            "How do vaccines work to provide immunity against diseases?",
            "What lifestyle changes help prevent heart disease?",
            "Can you compare different treatment approaches for cancer?",
            "How does stress affect the immune system function?",
            "What are the latest advances in medical technology?",
        ]
        
        # Combine all texts and create more samples by variation
        all_texts = []
        
        # Expand base texts with variations
        for _ in range(num_samples // (3 * (len(medical_texts) + len(general_texts) + len(mixed_texts)))):
            all_texts.extend(medical_texts)
            all_texts.extend(general_texts) 
            all_texts.extend(mixed_texts)
        
        # Ensure we have enough samples
        while len(all_texts) < num_samples:
            all_texts.extend(medical_texts[:10])
            all_texts.extend(general_texts[:10])
            all_texts.extend(mixed_texts[:10])
        
        all_texts = all_texts[:num_samples]
        
        # Split into train/eval (80/20 split)
        split_idx = int(0.8 * len(all_texts))
        train_texts = all_texts[:split_idx]
        eval_texts = all_texts[split_idx:]
        
        logger.info(f"Created datasets - Train: {len(train_texts)} samples, Eval: {len(eval_texts)} samples")
        
        return train_texts, eval_texts
    

    
    def setup_mol_runtime(self, config: MoLConfig) -> MoLRuntime:
        """Setup and configure MoL runtime with fusion layers."""
        
        logger.info("Setting up MoL Runtime...")
        
        # Initialize runtime
        mol_runtime = MoLRuntime(config)
        
        # Setup embeddings and LM head
        logger.info("📝 Setting up embeddings and LM head...")
        mol_runtime.setup_embeddings()
        mol_runtime.setup_lm_head()
        
        # Add 10 fusion layers with strategic layer selection
        logger.info("🔧 Adding 10 MoL fusion layers...")
        
        # Strategy: Gradually transition from general to medical expertise
        layer_mappings = [
            # Early layers: Both models contribute equally (general understanding)
            ([("Qwen/Qwen3-0.6B", 0), ("suayptalha/Qwen3-0.6B-Medical-Expert", 0)], 0),
            ([("Qwen/Qwen3-0.6B", 1), ("suayptalha/Qwen3-0.6B-Medical-Expert", 1)], 1),
            
            # Early-middle layers: Start specialization
            ([("Qwen/Qwen3-0.6B", 2), ("suayptalha/Qwen3-0.6B-Medical-Expert", 2)], 2),
            ([("Qwen/Qwen3-0.6B", 3), ("suayptalha/Qwen3-0.6B-Medical-Expert", 3)], 3),
            
            # Middle layers: Enhanced specialization
            ([("Qwen/Qwen3-0.6B", 4), ("suayptalha/Qwen3-0.6B-Medical-Expert", 4)], 4),
            ([("Qwen/Qwen3-0.6B", 5), ("suayptalha/Qwen3-0.6B-Medical-Expert", 5)], 5),
            
            # Late-middle layers: Cross-layer mixing for diversity
            ([("Qwen/Qwen3-0.6B", 6), ("suayptalha/Qwen3-0.6B-Medical-Expert", 7)], 6),
            ([("Qwen/Qwen3-0.6B", 7), ("suayptalha/Qwen3-0.6B-Medical-Expert", 6)], 7),
            
            # Final layers: High-level reasoning and output
            ([("Qwen/Qwen3-0.6B", 8), ("suayptalha/Qwen3-0.6B-Medical-Expert", 8)], 8),
            ([("Qwen/Qwen3-0.6B", 9), ("suayptalha/Qwen3-0.6B-Medical-Expert", 9)], 9),
        ]
        
        for layer_specs, layer_idx in layer_mappings:
            mol_runtime.add_layer(layer_specs, layer_idx)
            logger.info(f"  Added layer {layer_idx}: {layer_specs}")
        
        logger.info(f"✅ Added {len(mol_runtime.layers)} MoL fusion layers")
        
        # Print model statistics
        total_params = ModelUtils.count_parameters(mol_runtime)
        trainable_params = ModelUtils.count_parameters(mol_runtime, trainable_only=True)
        
        logger.info(f"📊 Model Statistics:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Trainable ratio: {trainable_params/total_params*100:.2f}%")
        
        self.training_metrics.update({
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_ratio": trainable_params/total_params
        })
        
        return mol_runtime
    
    def train(
        self, 
        epochs: int = 5,
        batch_size: int = 4,
        num_samples: int = 2000
    ) -> Dict[str, Any]:
        """Execute complete training pipeline."""
        
        self.start_time = time.time()
        logger.info("Starting Qwen Medical Fusion Training Pipeline...")
        
        try:
            # 1. Create configurations
            mol_config = self.create_mol_config()
            training_config = self.create_training_config(epochs, batch_size)
            
            # 2. Setup MoL runtime
            self.mol_runtime = self.setup_mol_runtime(mol_config)
            
            # 3. Create datasets
            train_texts, eval_texts = self.create_training_dataset(num_samples)
            
            # 4. Create tokenized datasets using create_simple_dataset
            train_dataset = create_simple_dataset(train_texts, self.mol_runtime.tokenizer, max_length=256)
            eval_dataset = create_simple_dataset(eval_texts, self.mol_runtime.tokenizer, max_length=256)
            
            # 5. Create data loaders
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=training_config.batch_size,
                shuffle=True,
                num_workers=0  # Avoid multiprocessing issues
            )
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=training_config.batch_size,
                shuffle=False,
                num_workers=0
            )
            
            logger.info(f"📚 Data loaded - Train batches: {len(train_dataloader)}, Eval batches: {len(eval_dataloader)}")
            
            # 6. Initialize trainer
            self.trainer = MoLTrainer(self.mol_runtime, training_config)
            
            # 7. Start training
            logger.info("🚀 Starting training...")
            training_results = self.trainer.train(train_dataloader, eval_dataloader)
            
            # 8. Save training results
            training_duration = time.time() - self.start_time
            self.training_metrics.update({
                "training_duration_seconds": training_duration,
                "training_results": training_results
            })
            
            # 9. Test the trained model
            self._test_model()
            
            # 10. Save final checkpoint
            self._save_final_checkpoint()
            
            # 11. Push to Hugging Face Hub if requested
            if self.push_to_hub and self.hub_repo_id:
                self._push_to_hub()
            
            logger.info("✅ Training pipeline completed successfully!")
            return self.training_metrics
            
        except Exception as e:
            logger.error(f"❌ Training pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _test_model(self):
        """Test the trained model with sample inputs."""
        
        logger.info("Testing trained model...")
        
        test_cases = [
            # General queries (should prefer general expert)
            "Hello, how are you today?",
            "What's the weather like?",
            "Tell me a joke.",
            
            # Medical queries (should prefer medical expert)
            "What are the symptoms of pneumonia?",
            "How is diabetes diagnosed?",
            "What causes high blood pressure?",
            
            # Mixed queries (should use intelligent routing)
            "Can you explain heart disease in simple terms?",
            "What lifestyle changes help prevent illness?",
        ]
        
        self.mol_runtime.eval()
        test_results = []
        
        for i, test_text in enumerate(test_cases):
            logger.info(f"  Test {i+1}: '{test_text[:50]}...'")
            
            try:
                inputs = self.mol_runtime.tokenizer(
                    test_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128
                )
                
                with torch.no_grad():
                    # Forward pass with routing stats
                    hidden_states, router_stats = self.mol_runtime.forward(
                        inputs['input_ids'],
                        inputs['attention_mask'],
                        return_router_stats=True
                    )
                    
                    # Analyze routing behavior
                    routing_analysis = self._analyze_routing_stats(router_stats)
                    test_results.append({
                        "input": test_text,
                        "routing_analysis": routing_analysis
                    })
                    
                    logger.info(f"    Routing: {routing_analysis}")
                
            except Exception as e:
                logger.warning(f"    Test failed: {e}")
        
        self.training_metrics["test_results"] = test_results
        logger.info("Model testing completed")
    
    def _analyze_routing_stats(self, router_stats: Dict) -> Dict:
        """Analyze routing statistics to understand expert usage."""
        
        if not router_stats:
            return {"error": "No routing stats available"}
        
        analysis = {
            "total_layers": len(router_stats),
            "expert_usage": {},
            "avg_entropy": 0.0,
            "routing_confidence": 0.0
        }
        
        total_entropy = 0.0
        total_confidence = 0.0
        
        for layer_idx, stats in router_stats.items():
            if 'expert_weights' in stats:
                expert_weights = stats['expert_weights']
                analysis["expert_usage"][f"layer_{layer_idx}"] = {
                    f"expert_{i}": float(weight) for i, weight in enumerate(expert_weights)
                }
            
            if 'router_entropy' in stats:
                entropy = stats['router_entropy']
                total_entropy += entropy
            
            if 'confidence' in stats:
                confidence = stats['confidence']
                total_confidence += confidence
        
        if len(router_stats) > 0:
            analysis["avg_entropy"] = total_entropy / len(router_stats)
            analysis["routing_confidence"] = total_confidence / len(router_stats)
        
        return analysis
    
    def _save_final_checkpoint(self):
        """Save final model checkpoint."""
        
        logger.info("Saving final checkpoint...")
        
        checkpoint_path = self.output_dir / "final_model"
        checkpoint_path.mkdir(exist_ok=True)
        
        # Save MoL runtime
        self.mol_runtime.save_checkpoint(str(checkpoint_path / "mol_runtime.pt"))
        
        # Save tokenizer
        self.mol_runtime.tokenizer.save_pretrained(checkpoint_path)
        
        # Save training metrics
        with open(checkpoint_path / "training_metrics.json", 'w') as f:
            json.dump(self.training_metrics, f, indent=2, default=str)
        
        # Save model info
        model_info = {
            "model_type": "MoL Fusion",
            "source_models": self.mol_runtime.config.models,
            "num_layers": len(self.mol_runtime.layers),
            "adapter_type": self.mol_runtime.config.adapter_type,
            "router_type": self.mol_runtime.config.router_type,
            "total_parameters": self.training_metrics.get("total_parameters", 0),
            "trainable_parameters": self.training_metrics.get("trainable_parameters", 0),
        }
        
        with open(checkpoint_path / "model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"Final checkpoint saved to: {checkpoint_path}")
    
    def _push_to_hub(self):
        """Push trained model to Hugging Face Hub."""
        
        logger.info("Pushing to Hugging Face Hub...")
        
        try:
            # Initialize HF publisher
            if self.hf_publisher is None:
                self.hf_publisher = HuggingFacePublisher()
            
            # Push MoL runtime (lightweight)
            runtime_repo_id = f"{self.hub_repo_id}-runtime"
            runtime_url = self.hf_publisher.push_mol_runtime(
                mol_runtime=self.mol_runtime,
                repo_id=runtime_repo_id,
                commit_message="Upload trained Qwen medical fusion MoL runtime",
                private=self.hub_private
            )
            
            # Push fused model (static)
            fused_repo_id = f"{self.hub_repo_id}-fused"
            fused_url = self.hf_publisher.push_fused_model(
                mol_runtime=self.mol_runtime,
                repo_id=fused_repo_id,
                commit_message="Upload fused Qwen medical expert model",
                private=self.hub_private,
                fusion_method="weighted_average"
            )
            
            self.training_metrics["hub_urls"] = {
                "runtime": runtime_url,
                "fused": fused_url
            }
            
            logger.info(f"Models pushed to Hub:")
            logger.info(f"  Runtime: {runtime_url}")
            logger.info(f"  Fused: {fused_url}")
            
        except Exception as e:
            logger.error(f"Hub push failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main training script entry point."""
    
    # Hardcoded configuration parameters
    epochs = 5
    batch_size = 4
    num_samples = 2000
    output_dir = "./qwen_fusion_output"
    hub_repo_id = "Abhaykoul/mango-test2"  # Set to your desired repo ID, e.g., "username/model-name"
    hub_private = False
    use_wandb = False
    push_to_hub = True
    
    # Configuration summary
    logger.info("Project Mango - Qwen Medical Expert Fusion")
    logger.info("=" * 60)
    logger.info(f"Models: Qwen/Qwen3-0.6B + suayptalha/Qwen3-0.6B-Medical-Expert")
    logger.info(f"Fusion Layers: 10")
    logger.info(f"Training Epochs: {epochs}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Training Samples: {num_samples}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Use W&B: {use_wandb}")
    logger.info(f"Push to Hub: {push_to_hub}")
    if hub_repo_id:
        logger.info(f"Hub Repo ID: {hub_repo_id}")
    logger.info("=" * 60)
    
    # Initialize trainer
    trainer = QwenMedicalFusionTrainer(
        output_dir=output_dir,
        use_wandb=use_wandb,
        push_to_hub=push_to_hub,
        hub_repo_id=hub_repo_id,
        hub_private=hub_private
    )
    
    # Execute training
    try:
        results = trainer.train(
            epochs=epochs,
            batch_size=batch_size,
            num_samples=num_samples
        )
        
        # Print final results
        logger.info("\nTraining Summary:")
        logger.info("=" * 40)
        logger.info(f"Total Parameters: {results.get('total_parameters', 'N/A'):,}")
        logger.info(f"Trainable Parameters: {results.get('trainable_parameters', 'N/A'):,}")
        logger.info(f"Training Duration: {results.get('training_duration_seconds', 0):.1f}s")
        
        if 'hub_urls' in results:
            logger.info("Hub URLs:")
            for model_type, url in results['hub_urls'].items():
                logger.info(f"  {model_type.title()}: {url}")
        
        logger.info("All done! Your Qwen medical fusion model is ready!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
