"""
Configuration validation for merge operations.
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigValidator:
    """
    Validator for merge configurations.
    """
    
    SUPPORTED_METHODS = ['slerp', 'ties', 'task_arithmetic', 'linear']
    SUPPORTED_DTYPES = ['float16', 'float32', 'bfloat16']
    SUPPORTED_DEVICES = ['cpu', 'cuda', 'auto']
    
    def __init__(self):
        pass
    
    def validate_config_dict(self, config_dict: Dict[str, Any]):
        """
        Validate configuration dictionary.
        
        Args:
            config_dict: Configuration dictionary from YAML
        """
        # Required fields
        if 'merge_method' not in config_dict:
            raise ValueError("Missing required field: 'merge_method'")
        
        # Validate merge method
        method = config_dict['merge_method']
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported merge method: {method}. "
                f"Supported methods: {self.SUPPORTED_METHODS}"
            )
        
        # Validate models or slices
        has_models = 'models' in config_dict and config_dict['models']
        has_slices = 'slices' in config_dict and config_dict['slices']
        
        if not has_models and not has_slices:
            raise ValueError("Must specify either 'models' or 'slices'")
        
        if has_models and has_slices:
            raise ValueError("Cannot specify both 'models' and 'slices'")
        
        # Validate slices if present
        if has_slices:
            self._validate_slices(config_dict['slices'])
        
        # Validate models if present
        if has_models:
            self._validate_models(config_dict['models'])
        
        # Validate optional fields
        if 'dtype' in config_dict:
            dtype = config_dict['dtype']
            if dtype not in self.SUPPORTED_DTYPES:
                raise ValueError(
                    f"Unsupported dtype: {dtype}. "
                    f"Supported dtypes: {self.SUPPORTED_DTYPES}"
                )
        
        if 'device' in config_dict:
            device = config_dict['device']
            if not self._is_valid_device(device):
                logger.warning(f"Device '{device}' may not be valid")
        
        # Validate parameters based on merge method
        if 'parameters' in config_dict:
            self._validate_parameters(config_dict['parameters'], method)
    
    def _validate_slices(self, slices: List[Dict[str, Any]]):
        """Validate slice configurations."""
        if not slices:
            raise ValueError("Slices list cannot be empty")
        
        for i, slice_config in enumerate(slices):
            if 'model' not in slice_config:
                raise ValueError(f"Slice {i}: Missing required field 'model'")
            
            if not isinstance(slice_config['model'], str):
                raise ValueError(f"Slice {i}: 'model' must be a string")
            
            # Validate layer_range if present
            if 'layer_range' in slice_config:
                layer_range = slice_config['layer_range']
                if not isinstance(layer_range, list) or len(layer_range) != 2:
                    raise ValueError(f"Slice {i}: 'layer_range' must be a list of 2 integers")
                
                if not all(isinstance(x, int) for x in layer_range):
                    raise ValueError(f"Slice {i}: 'layer_range' must contain integers")
                
                if layer_range[0] >= layer_range[1]:
                    raise ValueError(f"Slice {i}: layer_range start must be less than end")
                
                if layer_range[0] < 0:
                    raise ValueError(f"Slice {i}: layer_range start must be non-negative")
            
            # Validate weight if present
            if 'weight' in slice_config:
                weight = slice_config['weight']
                if not isinstance(weight, (int, float)):
                    raise ValueError(f"Slice {i}: 'weight' must be a number")
                
                if weight < 0:
                    raise ValueError(f"Slice {i}: 'weight' must be non-negative")
    
    def _validate_models(self, models: List[str]):
        """Validate model list."""
        if not models:
            raise ValueError("Models list cannot be empty")
        
        if len(models) < 2:
            raise ValueError("Need at least 2 models to merge")
        
        for i, model in enumerate(models):
            if not isinstance(model, str):
                raise ValueError(f"Model {i}: Must be a string")
            
            # Basic model name validation
            if not model.strip():
                raise ValueError(f"Model {i}: Cannot be empty or whitespace")
            
            # Check for common model naming patterns
            if not self._is_valid_model_name(model):
                logger.warning(f"Model {i} '{model}' may not be a valid HuggingFace model identifier")
    
    def _is_valid_model_name(self, model_name: str) -> bool:
        """Basic validation of model name format."""
        # Very basic checks for HuggingFace model naming
        if '/' in model_name:
            parts = model_name.split('/')
            if len(parts) != 2:
                return False
            org, name = parts
            if not org or not name:
                return False
        
        # Check for invalid characters
        invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
        if any(char in model_name for char in invalid_chars):
            return False
        
        return True
    
    def _validate_parameters(self, parameters: Dict[str, Any], method: str):
        """Validate parameters based on merge method."""
        if method == 'slerp':
            self._validate_slerp_parameters(parameters)
        elif method == 'ties':
            self._validate_ties_parameters(parameters)
        elif method == 'task_arithmetic':
            self._validate_task_arithmetic_parameters(parameters)
        elif method == 'linear':
            self._validate_linear_parameters(parameters)
    
    def _validate_slerp_parameters(self, params: Dict[str, Any]):
        """Validate SLERP parameters."""
        if 't' in params:
            t = params['t']
            if isinstance(t, (int, float)):
                if not (0 <= t <= 1):
                    raise ValueError("SLERP parameter 't' must be between 0 and 1")
            elif isinstance(t, list):
                if not all(isinstance(x, (int, float)) for x in t):
                    raise ValueError("SLERP parameter 't' list must contain numbers")
                if not all(0 <= x <= 1 for x in t):
                    raise ValueError("All SLERP 't' values must be between 0 and 1")
            elif isinstance(t, dict):
                # Layer-specific t values
                for key, value in t.items():
                    if isinstance(value, (int, float)):
                        if not (0 <= value <= 1):
                            raise ValueError(f"SLERP parameter 't.{key}' must be between 0 and 1")
                    elif isinstance(value, list):
                        if not all(isinstance(x, (int, float)) for x in value):
                            raise ValueError(f"SLERP parameter 't.{key}' list must contain numbers")
                        if not all(0 <= x <= 1 for x in value):
                            raise ValueError(f"All SLERP 't.{key}' values must be between 0 and 1")
        
        if 'eps' in params:
            eps = params['eps']
            if not isinstance(eps, (int, float)) or eps <= 0:
                raise ValueError("SLERP parameter 'eps' must be a positive number")
    
    def _validate_ties_parameters(self, params: Dict[str, Any]):
        """Validate TIES parameters."""
        if 'density' in params:
            density = params['density']
            if not isinstance(density, (int, float)) or not (0 < density <= 1):
                raise ValueError("TIES parameter 'density' must be between 0 and 1")
        
        if 'normalize' in params:
            if not isinstance(params['normalize'], bool):
                raise ValueError("TIES parameter 'normalize' must be a boolean")
        
        if 'majority_sign_method' in params:
            method = params['majority_sign_method']
            if method not in ['total', 'mass']:
                raise ValueError("TIES 'majority_sign_method' must be 'total' or 'mass'")
    
    def _validate_task_arithmetic_parameters(self, params: Dict[str, Any]):
        """Validate Task Arithmetic parameters."""
        if 'weights' in params:
            weights = params['weights']
            if not isinstance(weights, dict):
                raise ValueError("Task arithmetic 'weights' must be a dictionary")
            
            for model_name, weight in weights.items():
                if not isinstance(weight, (int, float)):
                    raise ValueError(f"Weight for model '{model_name}' must be a number")
                if weight < 0:
                    raise ValueError(f"Weight for model '{model_name}' must be non-negative")
        
        if 'scaling_factor' in params:
            factor = params['scaling_factor']
            if not isinstance(factor, (int, float)):
                raise ValueError("Task arithmetic 'scaling_factor' must be a number")
        
        if 'normalize' in params:
            if not isinstance(params['normalize'], bool):
                raise ValueError("Task arithmetic 'normalize' must be a boolean")
    
    def _validate_linear_parameters(self, params: Dict[str, Any]):
        """Validate Linear merge parameters."""
        if 'weights' in params:
            weights = params['weights']
            if not isinstance(weights, dict):
                raise ValueError("Linear merge 'weights' must be a dictionary")
            
            for model_name, weight in weights.items():
                if not isinstance(weight, (int, float)):
                    raise ValueError(f"Weight for model '{model_name}' must be a number")
                if weight < 0:
                    raise ValueError(f"Weight for model '{model_name}' must be non-negative")
        
        if 'normalize_weights' in params:
            if not isinstance(params['normalize_weights'], bool):
                raise ValueError("Linear merge 'normalize_weights' must be a boolean")
    
    def _is_valid_device(self, device: str) -> bool:
        """Check if device specification is valid."""
        if device in self.SUPPORTED_DEVICES:
            return True
        
        # Check for cuda:N format
        if device.startswith('cuda:'):
            try:
                int(device.split(':')[1])
                return True
            except (ValueError, IndexError):
                return False
        
        return False
    
    def validate_configuration(self, config):
        """
        Validate a MergeConfiguration object.
        
        Args:
            config: MergeConfiguration instance
        """
        # Validate merge method
        if config.merge_method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported merge method: {config.merge_method}. "
                f"Supported methods: {self.SUPPORTED_METHODS}"
            )
        
        # Validate that we have models to merge
        if not config.slices:
            raise ValueError("No model slices specified")
        
        if len(config.slices) < 2:
            raise ValueError("Need at least 2 model slices to merge")
        
        # Method-specific validation
        if config.merge_method == 'slerp' and len(config.slices) > 2:
            logger.warning("SLERP merge with more than 2 models will use hierarchical merging")
        
        if config.merge_method in ['ties', 'task_arithmetic']:
            if not config.base_model:
                raise ValueError(f"{config.merge_method} merge requires a base_model")
        
        # Validate output path
        if not config.output_path:
            raise ValueError("Output path cannot be empty")
        
        # Validate dtype
        if config.dtype not in self.SUPPORTED_DTYPES:
            raise ValueError(
                f"Unsupported dtype: {config.dtype}. "
                f"Supported dtypes: {self.SUPPORTED_DTYPES}"
            )
        
        logger.info(f"Configuration validation passed for {config.merge_method} merge")
    
    def get_validation_warnings(self, config) -> List[str]:
        """
        Get non-critical validation warnings.
        
        Args:
            config: MergeConfiguration instance
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        # Check for potential issues
        if config.merge_method == 'slerp' and len(config.slices) > 2:
            warnings.append("SLERP with >2 models will use hierarchical merging")
        
        if config.device == 'cuda' and config.dtype == 'float32':
            warnings.append("Using float32 on CUDA may require significant memory")
        
        if config.allow_crimes:
            warnings.append("'allow_crimes' is enabled - this may produce unexpected results")
        
        # Check for weight consistency
        if config.parameters and hasattr(config.parameters, 'weights'):
            model_names = config.get_model_list()
            weight_names = set(config.parameters.weights.keys())
            model_names_set = set(model_names)
            
            if weight_names != model_names_set:
                missing = model_names_set - weight_names
                extra = weight_names - model_names_set
                
                if missing:
                    warnings.append(f"No weights specified for models: {missing}")
                if extra:
                    warnings.append(f"Weights specified for unknown models: {extra}")
        
        return warnings