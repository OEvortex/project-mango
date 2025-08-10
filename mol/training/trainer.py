"""
Training pipeline for MoL system.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from dataclasses import dataclass
from tqdm import tqdm
import wandb
from pathlib import Path

from ..core.mol_runtime import MoLRuntime
from ..core.routers import compute_router_entropy, compute_load_balancing_loss
from ..utils.model_utils import ModelUtils

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for MoL training."""
    learning_rate: float = 1e-4
    router_learning_rate: float = 1e-5
    weight_decay: float = 0.01
    batch_size: int = 8
    max_epochs: int = 10
    max_steps: Optional[int] = None
    warmup_steps: int = 100
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    gradient_clip_norm: float = 1.0
    entropy_penalty_coeff: float = 0.1
    load_balancing_coeff: float = 0.01
    freeze_experts: bool = True
    use_gradient_checkpointing: bool = False
    output_dir: str = "./mol_checkpoints"
    run_name: Optional[str] = None
    use_wandb: bool = False


class MoLTrainer:
    """
    Training pipeline for MoL system.
    
    Handles training of adapters and routers while keeping expert models frozen.
    """
    
    def __init__(
        self,
        mol_runtime: MoLRuntime,
        config: TrainingConfig,
        tokenizer: Optional[Any] = None
    ):
        self.mol_runtime = mol_runtime
        self.config = config
        self.tokenizer = tokenizer or mol_runtime.tokenizer
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        
        # Initialize optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb if requested
        if config.use_wandb:
            wandb.init(
                project="mol-training",
                name=config.run_name,
                config=config.__dict__
            )
    
    def setup_optimization(self):
        """Setup optimizer and learning rate scheduler."""
        # Separate parameters for different learning rates
        adapter_params = []
        router_params = []
        
        for name, param in self.mol_runtime.named_parameters():
            if not param.requires_grad:
                continue
                
            if 'router' in name:
                router_params.append(param)
            else:
                adapter_params.append(param)
        
        # Create parameter groups with different learning rates
        param_groups = []
        if adapter_params:
            param_groups.append({
                'params': adapter_params,
                'lr': self.config.learning_rate,
                'weight_decay': self.config.weight_decay
            })
        
        if router_params:
            param_groups.append({
                'params': router_params,
                'lr': self.config.router_learning_rate,
                'weight_decay': self.config.weight_decay * 0.1  # Lower weight decay for routers
            })
        
        if not param_groups:
            raise ValueError("No trainable parameters found!")
        
        self.optimizer = optim.AdamW(param_groups)
        
        # Learning rate scheduler with warmup
        if self.config.warmup_steps > 0:
            from transformers import get_linear_schedule_with_warmup
            total_steps = self.config.max_steps or (self.config.max_epochs * 1000)  # Estimate
            
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=total_steps
            )
        
        logger.info(f"Setup optimizer with {len(param_groups)} parameter groups")
        logger.info(f"Trainable parameters: {ModelUtils.count_parameters(self.mol_runtime, trainable_only=True):,}")
    
    def freeze_experts(self):
        """Freeze expert model parameters."""
        for layer in self.mol_runtime.layers:
            for expert in layer.experts:
                ModelUtils.freeze_parameters(expert, freeze=True)
        
        logger.info("Frozen expert parameters")
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute training loss with regularization."""
        # Forward pass through MoL runtime
        hidden_states, router_stats = self.mol_runtime.forward(
            input_ids,
            attention_mask,
            return_router_stats=True
        )
        
        # Language modeling loss
        if self.mol_runtime.lm_head is None:
            raise RuntimeError("LM head not available. Cannot compute language modeling loss.")
        
        # Get logits from LM head
        logits = self.mol_runtime.lm_head(hidden_states)
        
        # Compute cross-entropy loss
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Router regularization losses
        total_entropy_loss = 0.0
        total_load_balance_loss = 0.0
        
        if router_stats:
            for layer_stats in router_stats.values():
                total_entropy_loss += layer_stats.get('router_entropy', 0.0)
                total_load_balance_loss += layer_stats.get('load_balancing_loss', 0.0)
            
            # Average across layers
            num_layers = len(router_stats)
            total_entropy_loss /= num_layers
            total_load_balance_loss /= num_layers
        
        # Entropy penalty (encourage exploration)
        entropy_penalty = -self.config.entropy_penalty_coeff * total_entropy_loss
        
        # Load balancing loss (encourage balanced expert usage)
        load_balance_loss = self.config.load_balancing_coeff * total_load_balance_loss
        
        # Total loss
        total_loss = lm_loss + entropy_penalty + load_balance_loss
        
        # Loss components for logging
        loss_components = {
            'lm_loss': lm_loss.item(),
            'entropy_penalty': entropy_penalty.item(),
            'load_balance_loss': load_balance_loss.item(),
            'total_loss': total_loss.item(),
            'router_entropy': total_entropy_loss,
        }
        
        return total_loss, loss_components
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Execute a single training step."""
        self.mol_runtime.train()
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch.get('labels', input_ids)  # Use input_ids as labels for causal LM
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass and loss computation
        loss, loss_components = self.compute_loss(input_ids, attention_mask, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.mol_runtime.parameters(),
                self.config.gradient_clip_norm
            )
        
        # Optimizer step
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        
        # Update global step
        self.global_step += 1
        
        # Add learning rate to loss components
        current_lr = self.optimizer.param_groups[0]['lr']
        loss_components['learning_rate'] = current_lr
        
        return loss_components
    
    def evaluate(
        self,
        eval_dataloader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate the model on validation data."""
        self.mol_runtime.eval()
        
        total_loss = 0.0
        total_lm_loss = 0.0
        total_entropy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch.get('labels', input_ids)
                
                loss, loss_components = self.compute_loss(input_ids, attention_mask, labels)
                
                total_loss += loss_components['total_loss']
                total_lm_loss += loss_components['lm_loss']
                total_entropy += loss_components['router_entropy']
                num_batches += 1
        
        eval_metrics = {
            'eval_loss': total_loss / num_batches,
            'eval_lm_loss': total_lm_loss / num_batches,
            'eval_router_entropy': total_entropy / num_batches,
        }
        
        return eval_metrics
    
    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None
    ):
        """Main training loop."""
        logger.info("Starting MoL training...")
        
        # Setup optimization
        self.setup_optimization()
        
        # Freeze experts if requested
        if self.config.freeze_experts:
            self.freeze_experts()
        
        # Enable gradient checkpointing if requested
        if self.config.use_gradient_checkpointing:
            self.mol_runtime.gradient_checkpointing_enable()
        
        # Training loop
        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch
            logger.info(f"Starting epoch {epoch + 1}/{self.config.max_epochs}")
            
            # Training phase
            self.mol_runtime.train()
            epoch_loss = 0.0
            num_batches = 0
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
            
            for batch in progress_bar:
                # Move batch to device
                device = next(self.mol_runtime.parameters()).device
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                
                # Training step
                loss_components = self.train_step(batch)
                epoch_loss += loss_components['total_loss']
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss_components['total_loss']:.4f}",
                    'lr': f"{loss_components['learning_rate']:.2e}",
                    'entropy': f"{loss_components['router_entropy']:.3f}"
                })
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    if self.config.use_wandb:
                        wandb.log(loss_components, step=self.global_step)
                    
                    logger.info(
                        f"Step {self.global_step}: "
                        f"loss={loss_components['total_loss']:.4f}, "
                        f"lm_loss={loss_components['lm_loss']:.4f}, "
                        f"entropy={loss_components['router_entropy']:.3f}"
                    )
                
                # Evaluation
                if eval_dataloader and self.global_step % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate(eval_dataloader)
                    
                    if self.config.use_wandb:
                        wandb.log(eval_metrics, step=self.global_step)
                    
                    logger.info(
                        f"Evaluation at step {self.global_step}: "
                        f"eval_loss={eval_metrics['eval_loss']:.4f}"
                    )
                    
                    # Save best model
                    if eval_metrics['eval_loss'] < self.best_loss:
                        self.best_loss = eval_metrics['eval_loss']
                        self.save_checkpoint(is_best=True)
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
                
                # Check if we've reached max steps
                if self.config.max_steps and self.global_step >= self.config.max_steps:
                    logger.info(f"Reached maximum steps ({self.config.max_steps})")
                    return
            
            # End of epoch logging
            avg_epoch_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Save final checkpoint
        self.save_checkpoint(is_final=True)
        logger.info("Training completed!")
    
    def save_checkpoint(self, is_best: bool = False, is_final: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'global_step': self.global_step,
            'epoch': self.current_epoch,
            'model_state_dict': self.mol_runtime.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Determine checkpoint filename
        if is_final:
            filename = "final_checkpoint.pt"
        elif is_best:
            filename = "best_checkpoint.pt"
        else:
            filename = f"checkpoint_step_{self.global_step}.pt"
        
        filepath = self.output_dir / filename
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        self.mol_runtime.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path} (step {self.global_step})")


def create_simple_dataset(texts: List[str], tokenizer, max_length: int = 512) -> torch.utils.data.Dataset:
    """Create a simple dataset from a list of texts."""
    
    class SimpleTextDataset(torch.utils.data.Dataset):
        def __init__(self, texts, tokenizer, max_length):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = self.texts[idx]
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
            }
    
    return SimpleTextDataset(texts, tokenizer, max_length)