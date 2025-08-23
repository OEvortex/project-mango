"""
Configuration data structures for merge operations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class ParameterConfig:
    """Configuration for merge parameters."""
    # SLERP parameters
    t: Union[float, List[float], Dict[str, Any]] = 0.5
    eps: float = 1e-8
    
    # TIES parameters  
    density: float = 0.8
    normalize: bool = True
    majority_sign_method: str = "total"
    
    # Task arithmetic parameters
    weights: Dict[str, float] = field(default_factory=dict)
    scaling_factor: float = 1.0
    
    # Linear merge parameters
    normalize_weights: bool = True
    
    # Layer-specific parameters
    self_attn: Optional[Dict[str, Any]] = None
    mlp: Optional[Dict[str, Any]] = None
    embed_tokens: Optional[Dict[str, Any]] = None
    lm_head: Optional[Dict[str, Any]] = None


@dataclass 
class SliceConfig:
    """Configuration for a model slice."""
    model: str
    layer_range: Optional[List[int]] = None
    parameters: Optional[ParameterConfig] = None
    weight: float = 1.0


@dataclass
class MergeConfiguration:
    """Main configuration for merge operations."""
    # Merge method (slerp, ties, task_arithmetic, linear)
    merge_method: str
    
    # Models to merge (for simple merges)
    models: Optional[List[str]] = None
    
    # Model slices (for complex merges)
    slices: Optional[List[SliceConfig]] = None
    
    # Base model (for task arithmetic, ties)
    base_model: Optional[str] = None
    
    # Global parameters
    parameters: Optional[ParameterConfig] = None
    
    # Output configuration
    dtype: str = "float16"
    output_path: str = "./merged_model"
    
    # Runtime configuration
    device: str = "cpu"
    lazy_unpickle: bool = True
    low_cpu_mem_usage: bool = True
    
    # Tokenizer configuration
    tokenizer_source: Optional[str] = None
    chat_template: Optional[str] = None
    
    # Advanced options
    allow_crimes: bool = False
    trust_remote_code: bool = False
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        if self.models is None and self.slices is None:
            raise ValueError("Either 'models' or 'slices' must be specified")
        
        if self.models is not None and self.slices is not None:
            raise ValueError("Cannot specify both 'models' and 'slices'")
        
        # Convert models to slices for uniform processing
        if self.models is not None:
            self.slices = [SliceConfig(model=model) for model in self.models]
            self.models = None
        
        # Set default parameters if not provided
        if self.parameters is None:
            self.parameters = ParameterConfig()
        
        # Set default base model
        if self.base_model is None and self.slices:
            self.base_model = self.slices[0].model
        
        # Set default tokenizer source
        if self.tokenizer_source is None:
            self.tokenizer_source = self.base_model
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'MergeConfiguration':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MergeConfiguration':
        """Create configuration from dictionary."""
        # Convert slices
        slices = None
        if 'slices' in config_dict:
            slices = []
            for slice_dict in config_dict['slices']:
                # Convert parameters if present
                params = None
                if 'parameters' in slice_dict:
                    params = ParameterConfig(**slice_dict['parameters'])
                
                slice_config = SliceConfig(
                    model=slice_dict['model'],
                    layer_range=slice_dict.get('layer_range'),
                    parameters=params,
                    weight=slice_dict.get('weight', 1.0)
                )
                slices.append(slice_config)
        
        # Convert global parameters
        parameters = None
        if 'parameters' in config_dict:
            parameters = ParameterConfig(**config_dict['parameters'])
        
        # Create main config
        config = cls(
            merge_method=config_dict['merge_method'],
            models=config_dict.get('models'),
            slices=slices,
            base_model=config_dict.get('base_model'),
            parameters=parameters,
            dtype=config_dict.get('dtype', 'float16'),
            output_path=config_dict.get('output_path', './merged_model'),
            device=config_dict.get('device', 'cpu'),
            lazy_unpickle=config_dict.get('lazy_unpickle', True),
            low_cpu_mem_usage=config_dict.get('low_cpu_mem_usage', True),
            tokenizer_source=config_dict.get('tokenizer_source'),
            chat_template=config_dict.get('chat_template'),
            allow_crimes=config_dict.get('allow_crimes', False),
            trust_remote_code=config_dict.get('trust_remote_code', False)
        )
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {
            'merge_method': self.merge_method,
            'base_model': self.base_model,
            'dtype': self.dtype,
            'output_path': self.output_path,
            'device': self.device,
            'lazy_unpickle': self.lazy_unpickle,
            'low_cpu_mem_usage': self.low_cpu_mem_usage,
        }
        
        # Add optional fields
        if self.tokenizer_source:
            result['tokenizer_source'] = self.tokenizer_source
        if self.chat_template:
            result['chat_template'] = self.chat_template
        if self.allow_crimes:
            result['allow_crimes'] = self.allow_crimes
        if self.trust_remote_code:
            result['trust_remote_code'] = self.trust_remote_code
        
        # Add slices
        if self.slices:
            result['slices'] = []
            for slice_config in self.slices:
                slice_dict = {
                    'model': slice_config.model,
                    'weight': slice_config.weight,
                }
                
                if slice_config.layer_range:
                    slice_dict['layer_range'] = slice_config.layer_range
                
                if slice_config.parameters:
                    slice_dict['parameters'] = slice_config.parameters.__dict__
                
                result['slices'].append(slice_dict)
        
        # Add global parameters
        if self.parameters:
            result['parameters'] = self.parameters.__dict__
        
        return result
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        config_dict = self.to_dict()
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def get_model_list(self) -> List[str]:
        """Get list of all models involved in the merge."""
        if self.slices:
            return [slice_config.model for slice_config in self.slices]
        return []
    
    def get_slice_for_model(self, model_name: str) -> Optional[SliceConfig]:
        """Get slice configuration for a specific model."""
        if self.slices:
            for slice_config in self.slices:
                if slice_config.model == model_name:
                    return slice_config
        return None
    
    def get_effective_parameters(self, model_name: str) -> ParameterConfig:
        """Get effective parameters for a model (combines global and model-specific)."""
        # Start with global parameters
        effective_params = ParameterConfig()
        if self.parameters:
            for key, value in self.parameters.__dict__.items():
                setattr(effective_params, key, value)
        
        # Override with model-specific parameters
        slice_config = self.get_slice_for_model(model_name)
        if slice_config and slice_config.parameters:
            for key, value in slice_config.parameters.__dict__.items():
                if value is not None:
                    setattr(effective_params, key, value)
        
        return effective_params