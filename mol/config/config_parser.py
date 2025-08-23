"""
Configuration parser for YAML merge configurations.
"""

import yaml
import os
from typing import Dict, Any, List, Optional
import logging

from .merge_config import MergeConfiguration
from .validation import ConfigValidator

logger = logging.getLogger(__name__)


class ConfigParser:
    """
    Parser for merge configuration files.
    """
    
    def __init__(self):
        self.validator = ConfigValidator()
    
    def parse_config(self, config_path: str) -> MergeConfiguration:
        """
        Parse configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Parsed merge configuration
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        logger.info(f"Parsing configuration from {config_path}")
        
        # Load YAML
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}")
        except Exception as e:
            raise ValueError(f"Error reading configuration file: {e}")
        
        # Validate configuration
        self.validator.validate_config_dict(config_dict)
        
        # Parse into configuration object
        config = MergeConfiguration.from_dict(config_dict)
        
        # Final validation
        self.validator.validate_configuration(config)
        
        logger.info(f"Successfully parsed configuration for {config.merge_method} merge")
        return config
    
    def parse_config_string(self, config_string: str) -> MergeConfiguration:
        """
        Parse configuration from YAML string.
        
        Args:
            config_string: YAML configuration as string
            
        Returns:
            Parsed merge configuration
        """
        try:
            config_dict = yaml.safe_load(config_string)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}")
        
        # Validate and parse
        self.validator.validate_config_dict(config_dict)
        config = MergeConfiguration.from_dict(config_dict)
        self.validator.validate_configuration(config)
        
        return config
    
    def create_example_configs(self, output_dir: str = "./examples"):
        """Create example configuration files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # SLERP example
        slerp_config = {
            'merge_method': 'slerp',
            'slices': [
                {
                    'model': 'microsoft/DialoGPT-medium',
                    'layer_range': [0, 24]
                },
                {
                    'model': 'microsoft/DialoGPT-large', 
                    'layer_range': [0, 24]
                }
            ],
            'base_model': 'microsoft/DialoGPT-medium',
            'parameters': {
                't': 0.5,
                'eps': 1e-8
            },
            'dtype': 'float16',
            'output_path': './slerp_merged'
        }
        
        with open(os.path.join(output_dir, 'slerp_example.yml'), 'w') as f:
            yaml.dump(slerp_config, f, default_flow_style=False, indent=2)
        
        # TIES example
        ties_config = {
            'merge_method': 'ties',
            'slices': [
                {
                    'model': 'microsoft/DialoGPT-medium',
                    'layer_range': [0, 24],
                    'weight': 1.0
                },
                {
                    'model': 'microsoft/DialoGPT-large',
                    'layer_range': [0, 24], 
                    'weight': 1.0
                },
                {
                    'model': 'gpt2',
                    'layer_range': [0, 12],
                    'weight': 0.5
                }
            ],
            'base_model': 'gpt2',
            'parameters': {
                'density': 0.8,
                'normalize': True,
                'majority_sign_method': 'total'
            },
            'dtype': 'float16',
            'output_path': './ties_merged'
        }
        
        with open(os.path.join(output_dir, 'ties_example.yml'), 'w') as f:
            yaml.dump(ties_config, f, default_flow_style=False, indent=2)
        
        # Task Arithmetic example
        task_config = {
            'merge_method': 'task_arithmetic',
            'slices': [
                {
                    'model': 'microsoft/DialoGPT-medium',
                    'weight': 1.0
                },
                {
                    'model': 'microsoft/DialoGPT-large',
                    'weight': 0.7
                }
            ],
            'base_model': 'gpt2',
            'parameters': {
                'weights': {
                    'microsoft/DialoGPT-medium': 1.0,
                    'microsoft/DialoGPT-large': 0.7
                },
                'normalize': False,
                'scaling_factor': 1.0
            },
            'dtype': 'float16',
            'output_path': './task_arithmetic_merged'
        }
        
        with open(os.path.join(output_dir, 'task_arithmetic_example.yml'), 'w') as f:
            yaml.dump(task_config, f, default_flow_style=False, indent=2)
        
        # Linear merge example
        linear_config = {
            'merge_method': 'linear',
            'slices': [
                {
                    'model': 'gpt2',
                    'weight': 0.6
                },
                {
                    'model': 'microsoft/DialoGPT-medium',
                    'weight': 0.4
                }
            ],
            'parameters': {
                'weights': {
                    'gpt2': 0.6,
                    'microsoft/DialoGPT-medium': 0.4
                },
                'normalize_weights': True
            },
            'dtype': 'float16',
            'output_path': './linear_merged'
        }
        
        with open(os.path.join(output_dir, 'linear_example.yml'), 'w') as f:
            yaml.dump(linear_config, f, default_flow_style=False, indent=2)
        
        # Complex gradient SLERP example
        gradient_slerp_config = {
            'merge_method': 'slerp',
            'slices': [
                {
                    'model': 'gpt2',
                    'layer_range': [0, 12]
                },
                {
                    'model': 'microsoft/DialoGPT-medium',
                    'layer_range': [0, 12]
                }
            ],
            'base_model': 'gpt2',
            'parameters': {
                't': [0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2, 0, 0],
                'self_attn': {
                    't': [0, 0.5, 0.3, 0.7, 1]
                },
                'mlp': {
                    't': [1, 0.5, 0.7, 0.3, 0]
                }
            },
            'dtype': 'float16',
            'output_path': './gradient_slerp_merged'
        }
        
        with open(os.path.join(output_dir, 'gradient_slerp_example.yml'), 'w') as f:
            yaml.dump(gradient_slerp_config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Created example configurations in {output_dir}")
    
    def validate_file_syntax(self, config_path: str) -> bool:
        """
        Validate YAML syntax without full parsing.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            True if syntax is valid
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml.safe_load(f)
            return True
        except yaml.YAMLError as e:
            logger.error(f"YAML syntax error: {e}")
            return False
        except Exception as e:
            logger.error(f"File error: {e}")
            return False
    
    def get_config_summary(self, config: MergeConfiguration) -> str:
        """
        Get a summary of the configuration.
        
        Args:
            config: Merge configuration
            
        Returns:
            Human-readable summary
        """
        summary = []
        summary.append(f"Merge Method: {config.merge_method}")
        summary.append(f"Base Model: {config.base_model}")
        summary.append(f"Output Path: {config.output_path}")
        summary.append(f"Data Type: {config.dtype}")
        summary.append(f"Device: {config.device}")
        
        if config.slices:
            summary.append(f"Models to merge ({len(config.slices)}):")
            for i, slice_config in enumerate(config.slices):
                layer_info = ""
                if slice_config.layer_range:
                    layer_info = f" (layers {slice_config.layer_range[0]}-{slice_config.layer_range[1]})"
                summary.append(f"  {i+1}. {slice_config.model}{layer_info} (weight: {slice_config.weight})")
        
        if config.parameters:
            summary.append("Parameters:")
            params_dict = config.parameters.__dict__
            for key, value in params_dict.items():
                if value is not None:
                    if isinstance(value, dict) and not value:
                        continue
                    summary.append(f"  {key}: {value}")
        
        return "\n".join(summary)