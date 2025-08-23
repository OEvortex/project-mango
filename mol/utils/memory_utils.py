"""
Memory management utilities for MoL system.
"""

import torch
import gc
import psutil
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_ram: float  # GB
    available_ram: float  # GB
    used_ram: float  # GB
    total_vram: float  # GB (if GPU available)
    used_vram: float  # GB (if GPU available)
    available_vram: float  # GB (if GPU available)


class MemoryManager:
    """
    Memory management for MoL system.
    
    Handles lazy loading, offloading, and memory optimization
    for large model combinations.
    """
    
    def __init__(self):
        self.device_map: Dict[str, str] = {}
        self.offloaded_modules: Dict[str, torch.nn.Module] = {}
        self.memory_threshold = 0.8  # Offload when memory usage > 80%
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory usage statistics."""
        # RAM statistics
        ram = psutil.virtual_memory()
        total_ram = ram.total / (1024**3)  # GB
        available_ram = ram.available / (1024**3)  # GB
        used_ram = (ram.total - ram.available) / (1024**3)  # GB
        
        # VRAM statistics
        total_vram = 0.0
        used_vram = 0.0
        available_vram = 0.0
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                mem_info = torch.cuda.mem_get_info(i)
                
                device_total = device_props.total_memory / (1024**3)  # GB
                device_free = mem_info[0] / (1024**3)  # GB
                device_used = device_total - device_free
                
                total_vram += device_total
                used_vram += device_used
                available_vram += device_free
        
        return MemoryStats(
            total_ram=total_ram,
            available_ram=available_ram,
            used_ram=used_ram,
            total_vram=total_vram,
            used_vram=used_vram,
            available_vram=available_vram
        )
    
    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure."""
        stats = self.get_memory_stats()
        
        # Check RAM pressure
        ram_usage = stats.used_ram / stats.total_ram if stats.total_ram > 0 else 0
        
        # Check VRAM pressure if GPU available
        vram_usage = 0
        if stats.total_vram > 0:
            vram_usage = stats.used_vram / stats.total_vram
        
        memory_pressure = max(ram_usage, vram_usage) > self.memory_threshold
        
        if memory_pressure:
            logger.warning(
                f"Memory pressure detected: RAM {ram_usage:.1%}, "
                f"VRAM {vram_usage:.1%}"
            )
        
        return memory_pressure
    
    def optimize_memory(self):
        """Optimize memory usage by clearing caches and garbage collection."""
        # Clear Python garbage
        gc.collect()
        
        # Clear PyTorch caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Clear unused variables
        for obj in gc.get_objects():
            if torch.is_tensor(obj):
                if obj.grad is not None:
                    obj.grad = None
        
        logger.info("Memory optimization completed")
    
    def offload_module(
        self, 
        module: torch.nn.Module, 
        module_name: str,
        target_device: str = "cpu"
    ):
        """Offload a module to specified device (usually CPU)."""
        if module_name in self.offloaded_modules:
            logger.warning(f"Module {module_name} already offloaded")
            return
        
        # Store original device
        original_device = next(module.parameters()).device
        
        # Move to target device
        module.to(target_device)
        
        # Store reference
        self.offloaded_modules[module_name] = module
        self.device_map[module_name] = target_device
        
        logger.info(f"Offloaded {module_name} from {original_device} to {target_device}")
    
    def load_module(
        self, 
        module_name: str, 
        target_device: str = "cuda"
    ) -> Optional[torch.nn.Module]:
        """Load a previously offloaded module to specified device."""
        if module_name not in self.offloaded_modules:
            logger.warning(f"Module {module_name} not found in offloaded modules")
            return None
        
        module = self.offloaded_modules[module_name]
        
        # Check memory before loading
        if self.check_memory_pressure():
            logger.warning(f"Cannot load {module_name}: memory pressure detected")
            return None
        
        # Move to target device
        module.to(target_device)
        self.device_map[module_name] = target_device
        
        logger.info(f"Loaded {module_name} to {target_device}")
        return module
    
    @contextmanager
    def temporary_load(self, module_name: str, target_device: str = "cuda"):
        """Context manager for temporarily loading an offloaded module."""
        module = self.load_module(module_name, target_device)
        try:
            yield module
        finally:
            if module is not None:
                # Offload back to CPU
                self.offload_module(module, module_name, "cpu")
    
    def smart_device_placement(
        self,
        models: Dict[str, Any],
        target_device: str = "cuda",
        memory_limit_gb: float = None
    ) -> Dict[str, str]:
        """
        Intelligently place models across devices based on memory constraints.
        
        Args:
            models: Dictionary of models to place
            target_device: Preferred device
            memory_limit_gb: Memory limit in GB per device
            
        Returns:
            Device placement mapping
        """
        device_map = {}
        
        if memory_limit_gb is None:
            stats = self.get_memory_stats()
            if target_device.startswith("cuda") and stats.total_vram > 0:
                memory_limit_gb = stats.available_vram * 0.8  # Use 80% of available
            else:
                memory_limit_gb = stats.available_ram * 0.6   # Use 60% of available
        
        current_memory_usage = 0.0
        
        for model_name, model in models.items():
            # Estimate model memory
            model_memory = self.estimate_model_memory(model)
            
            # Check if we can fit on target device
            if current_memory_usage + model_memory <= memory_limit_gb:
                device_map[model_name] = target_device
                current_memory_usage += model_memory
            else:
                # Fallback to CPU
                device_map[model_name] = "cpu"
                logger.warning(
                    f"Model {model_name} placed on CPU due to memory constraints "
                    f"({model_memory:.1f}GB needed, {memory_limit_gb - current_memory_usage:.1f}GB available)"
                )
        
        return device_map
    
    def lazy_load_model(
        self,
        model_name: str,
        model_loader_func,
        cache_key: str = None
    ) -> Any:
        """
        Lazy load model with caching and memory management.
        
        Args:
            model_name: Name/identifier of the model
            model_loader_func: Function to load the model
            cache_key: Optional cache key (defaults to model_name)
            
        Returns:
            Loaded model
        """
        if cache_key is None:
            cache_key = model_name
        
        # Check if already loaded
        if cache_key in self.offloaded_modules:
            logger.info(f"Loading cached model: {cache_key}")
            return self.offloaded_modules[cache_key]
        
        # Check memory pressure before loading
        if self.check_memory_pressure():
            logger.warning(f"Memory pressure detected before loading {model_name}")
            self.optimize_memory()
        
        # Load model
        logger.info(f"Lazy loading model: {model_name}")
        try:
            model = model_loader_func()
            
            # Cache the model
            self.offloaded_modules[cache_key] = model
            
            # Check memory after loading
            if self.check_memory_pressure():
                logger.warning(f"Memory pressure detected after loading {model_name}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to lazy load {model_name}: {e}")
            raise
    
    def batch_offload(
        self,
        modules: Dict[str, Any],
        target_device: str = "cpu",
        keep_top_k: int = 2
    ):
        """
        Batch offload multiple modules, keeping top-k most recently used.
        
        Args:
            modules: Dictionary of modules to potentially offload
            target_device: Device to offload to
            keep_top_k: Number of modules to keep on current device
        """
        if len(modules) <= keep_top_k:
            logger.info(f"Keeping all {len(modules)} modules (under keep_top_k={keep_top_k})")
            return
        
        # Sort by some criteria (e.g., size, last used time)
        # For now, keep first keep_top_k modules
        modules_to_offload = list(modules.items())[keep_top_k:]
        
        logger.info(f"Batch offloading {len(modules_to_offload)} modules to {target_device}")
        
        for name, module in modules_to_offload:
            try:
                self.offload_module(module, name, target_device)
            except Exception as e:
                logger.warning(f"Failed to offload {name}: {e}")
    
    def adaptive_memory_management(
        self,
        operation_func,
        *args,
        memory_threshold: float = 0.85,
        max_retries: int = 3,
        **kwargs
    ):
        """
        Execute operation with adaptive memory management.
        
        Args:
            operation_func: Function to execute
            *args: Arguments for the function
            memory_threshold: Memory threshold to trigger cleanup
            max_retries: Maximum number of retries
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of operation_func
        """
        for attempt in range(max_retries + 1):
            try:
                # Check memory before operation
                stats = self.get_memory_stats()
                memory_usage = max(
                    stats.used_ram / stats.total_ram if stats.total_ram > 0 else 0,
                    stats.used_vram / stats.total_vram if stats.total_vram > 0 else 0
                )
                
                if memory_usage > memory_threshold:
                    logger.warning(
                        f"Memory usage {memory_usage:.1%} exceeds threshold {memory_threshold:.1%}"
                    )
                    self.optimize_memory()
                
                # Execute operation
                result = operation_func(*args, **kwargs)
                return result
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and attempt < max_retries:
                    logger.warning(f"OOM error on attempt {attempt + 1}, cleaning up memory")
                    self.optimize_memory()
                    
                    # More aggressive cleanup on later attempts
                    if attempt > 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        
                        # Clear more cached items
                        for obj in list(self.offloaded_modules.keys()):
                            if len(self.offloaded_modules) > 1:  # Keep at least one
                                del self.offloaded_modules[obj]
                                break
                else:
                    raise
        
        raise RuntimeError(f"Operation failed after {max_retries + 1} attempts")
    
    def estimate_model_memory(
        self, 
        model: torch.nn.Module, 
        dtype: torch.dtype = torch.float16
    ) -> float:
        """Estimate memory usage of a model in GB."""
        total_params = sum(p.numel() for p in model.parameters())
        
        # Bytes per parameter based on dtype
        bytes_per_param = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.int8: 1,
        }.get(dtype, 4)
        
        # Estimate memory (parameters + gradients + optimizer states + activations)
        param_memory = total_params * bytes_per_param
        gradient_memory = total_params * bytes_per_param  # If training
        optimizer_memory = total_params * bytes_per_param * 2  # Adam states
        activation_memory = param_memory * 0.5  # Rough estimate
        
        total_memory = param_memory + gradient_memory + optimizer_memory + activation_memory
        return total_memory / (1024**3)  # Convert to GB
    
    def suggest_batch_size(
        self, 
        model_memory: float,
        sequence_length: int = 512,
        hidden_dim: int = 768,
        available_memory: float = None
    ) -> int:
        """Suggest optimal batch size based on available memory."""
        if available_memory is None:
            stats = self.get_memory_stats()
            available_memory = min(stats.available_ram, stats.available_vram)
        
        # Estimate memory per sample (very rough approximation)
        memory_per_sample = (
            sequence_length * hidden_dim * 4 +  # Hidden states (float32)
            sequence_length * 4 +  # Attention scores
            model_memory * 0.1  # Rough overhead
        ) / (1024**3)  # Convert to GB
        
        # Leave some memory for safety
        usable_memory = available_memory * 0.7
        
        suggested_batch_size = max(1, int(usable_memory / memory_per_sample))
        
        logger.info(
            f"Suggested batch size: {suggested_batch_size} "
            f"(available memory: {available_memory:.1f}GB, "
            f"estimated per sample: {memory_per_sample*1000:.1f}MB)"
        )
        
        return suggested_batch_size