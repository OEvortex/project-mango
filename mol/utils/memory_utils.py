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
    
    def create_device_map(
        self, 
        modules: List[str], 
        available_devices: List[str],
        memory_per_module: Optional[Dict[str, float]] = None
    ) -> Dict[str, str]:
        """
        Create device mapping for modules based on memory constraints.
        
        Args:
            modules: List of module names
            available_devices: List of available devices
            memory_per_module: Estimated memory usage per module (GB)
        
        Returns:
            Device mapping dictionary
        """
        device_map = {}
        device_memory = {}
        
        # Get device memory info
        for device in available_devices:
            if device == "cpu":
                device_memory[device] = psutil.virtual_memory().available / (1024**3)
            elif device.startswith("cuda"):
                device_idx = int(device.split(":")[-1]) if ":" in device else 0
                if torch.cuda.is_available() and device_idx < torch.cuda.device_count():
                    mem_info = torch.cuda.mem_get_info(device_idx)
                    device_memory[device] = mem_info[0] / (1024**3)  # Available memory
                else:
                    device_memory[device] = 0.0
        
        # Sort devices by available memory (descending)
        sorted_devices = sorted(device_memory.items(), key=lambda x: x[1], reverse=True)
        
        # Assign modules to devices
        current_device_idx = 0
        current_device_memory = {dev: mem for dev, mem in sorted_devices}
        
        for module in modules:
            # Get memory requirement for this module
            module_memory = memory_per_module.get(module, 1.0) if memory_per_module else 1.0
            
            # Find device with enough memory
            assigned = False
            for i, (device, available_mem) in enumerate(sorted_devices):
                if current_device_memory[device] >= module_memory:
                    device_map[module] = device
                    current_device_memory[device] -= module_memory
                    assigned = True
                    break
            
            # If no device has enough memory, assign to CPU
            if not assigned:
                device_map[module] = "cpu"
                logger.warning(
                    f"Module {module} assigned to CPU due to memory constraints "
                    f"(requires {module_memory:.1f}GB)"
                )
        
        return device_map
    
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