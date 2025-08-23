"""
CLI utilities for configuration validation.
"""

import sys
from typing import List

from ..config import ConfigParser, ConfigValidator


def validate_command(args):
    """Validate configuration file."""
    print(f"🔍 Validating configuration: {args.config}")
    
    parser = ConfigParser()
    validator = ConfigValidator()
    
    # Check syntax
    if not parser.validate_file_syntax(args.config):
        print("❌ Configuration file has syntax errors")
        return 1
    
    # Parse and validate
    try:
        config = parser.parse_config(args.config)
        print("✅ Configuration is valid")
        
        # Show summary
        print("\n📋 Configuration Summary:")
        print(parser.get_config_summary(config))
        
        # Show warnings
        warnings = validator.get_validation_warnings(config)
        if warnings:
            print("\n⚠️  Warnings:")
            for warning in warnings:
                print(f"  - {warning}")
        
        # Additional validation info
        if args.verbose:
            print(f"\n🔧 Additional Details:")
            print(f"  - Merge method: {config.merge_method}")
            print(f"  - Number of models: {len(config.get_model_list())}")
            print(f"  - Base model: {config.base_model}")
            print(f"  - Output dtype: {config.dtype}")
            print(f"  - Target device: {config.device}")
            
            if config.parameters:
                print(f"  - Has parameters: Yes")
                params = config.parameters.__dict__
                non_none_params = {k: v for k, v in params.items() if v is not None}
                if non_none_params:
                    print(f"  - Parameter count: {len(non_none_params)}")
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return 1
    
    return 0


def batch_validate(config_files: List[str], verbose: bool = False) -> int:
    """Validate multiple configuration files."""
    print(f"🔍 Batch validating {len(config_files)} configuration files")
    
    parser = ConfigParser()
    validator = ConfigValidator()
    
    results = []
    for config_file in config_files:
        try:
            # Check syntax
            if not parser.validate_file_syntax(config_file):
                results.append((config_file, False, "Syntax error"))
                continue
            
            # Parse and validate
            config = parser.parse_config(config_file)
            warnings = validator.get_validation_warnings(config)
            
            if warnings and verbose:
                results.append((config_file, True, f"Valid (with {len(warnings)} warnings)"))
            else:
                results.append((config_file, True, "Valid"))
                
        except Exception as e:
            results.append((config_file, False, str(e)))
    
    # Print results
    print("\n📊 Validation Results:")
    valid_count = 0
    for config_file, is_valid, message in results:
        status = "✅" if is_valid else "❌"
        print(f"  {status} {config_file}: {message}")
        if is_valid:
            valid_count += 1
    
    print(f"\n📈 Summary: {valid_count}/{len(config_files)} files valid")
    
    return 0 if valid_count == len(config_files) else 1


def check_model_compatibility(config_file: str) -> int:
    """Check if models in configuration are compatible for merging."""
    print(f"🔍 Checking model compatibility for: {config_file}")
    
    parser = ConfigParser()
    
    try:
        config = parser.parse_config(config_file)
        model_list = config.get_model_list()
        
        print(f"📋 Models to check ({len(model_list)}):")
        for i, model in enumerate(model_list):
            print(f"  {i+1}. {model}")
        
        # Try to load model configs to check compatibility
        from transformers import AutoConfig
        
        configs = {}
        for model in model_list:
            try:
                model_config = AutoConfig.from_pretrained(model)
                configs[model] = model_config
                print(f"✅ {model}: Loaded config")
            except Exception as e:
                print(f"❌ {model}: Failed to load config - {e}")
                return 1
        
        # Check compatibility
        if len(configs) >= 2:
            first_model = list(configs.keys())[0]
            first_config = configs[first_model]
            
            print(f"\n🔧 Compatibility Analysis (vs {first_model}):")
            
            compatible = True
            for model, model_config in configs.items():
                if model == first_model:
                    continue
                
                issues = []
                
                # Check hidden size
                if model_config.hidden_size != first_config.hidden_size:
                    issues.append(f"hidden_size: {model_config.hidden_size} vs {first_config.hidden_size}")
                
                # Check number of layers
                if model_config.num_hidden_layers != first_config.num_hidden_layers:
                    issues.append(f"num_layers: {model_config.num_hidden_layers} vs {first_config.num_hidden_layers}")
                
                # Check attention heads
                if hasattr(model_config, 'num_attention_heads') and hasattr(first_config, 'num_attention_heads'):
                    if model_config.num_attention_heads != first_config.num_attention_heads:
                        issues.append(f"attention_heads: {model_config.num_attention_heads} vs {first_config.num_attention_heads}")
                
                if issues:
                    print(f"⚠️  {model}: {', '.join(issues)}")
                    if config.merge_method in ['slerp', 'linear'] and any('hidden_size' in issue for issue in issues):
                        compatible = False
                else:
                    print(f"✅ {model}: Compatible")
            
            if compatible:
                print(f"\n✅ All models are compatible for {config.merge_method} merge")
            else:
                print(f"\n❌ Models have compatibility issues for {config.merge_method} merge")
                print("   Consider using a merge method that supports dimension adaptation")
                return 1
        
    except Exception as e:
        print(f"❌ Error checking compatibility: {e}")
        return 1
    
    return 0