"""
Main CLI for MoL merge operations.
"""

import argparse
import sys
import os
import logging
from typing import Optional
import torch

from ..config import ConfigParser, ConfigValidator
from ..merge_methods import SlerpMerge, TiesMerge, TaskArithmeticMerge, LinearMerge
from ..merge_methods.base_merge import MergeConfig as BaseMergeConfig
from ..utils.memory_utils import MemoryManager
from ..utils.model_utils import ModelUtils

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def merge_command(args):
    """Execute merge operation."""
    print(f"🔄 MoL Merge - Starting {args.config} merge operation")
    
    # Security warning for trust_remote_code
    if getattr(args, 'trust_remote_code', False):
        print("⚠️  WARNING: trust_remote_code=True enabled!")
        print("   This may execute arbitrary code from model repositories.")
        print("   Only proceed if you trust all models in the configuration.")
        print()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Parse configuration
    parser = ConfigParser()
    try:
        config = parser.parse_config(args.config)
        print(f"✅ Loaded configuration: {config.merge_method} merge")
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return 1
    
    # Print configuration summary
    if args.verbose:
        print("\n📋 Configuration Summary:")
        print(parser.get_config_summary(config))
        print()
    
    # Override configuration with CLI arguments
    if args.output_path:
        config.output_path = args.output_path
    if args.device:
        config.device = args.device
    if args.dtype:
        config.dtype = args.dtype
    if args.low_cpu_mem_usage is not None:
        config.low_cpu_mem_usage = args.low_cpu_mem_usage
    if args.lazy_unpickle is not None:
        config.lazy_unpickle = args.lazy_unpickle
    if hasattr(args, 'trust_remote_code') and args.trust_remote_code is not None:
        config.trust_remote_code = args.trust_remote_code
    
    # Check device availability
    if config.device.startswith('cuda') and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available, falling back to CPU")
        config.device = 'cpu'
    
    # Memory management
    memory_manager = MemoryManager()
    if args.verbose:
        stats = memory_manager.get_memory_stats()
        print(f"💾 Memory - RAM: {stats.used_ram:.1f}/{stats.total_ram:.1f} GB, ", end="")
        if stats.total_vram > 0:
            print(f"VRAM: {stats.used_vram:.1f}/{stats.total_vram:.1f} GB")
        else:
            print("No GPU available")
    
    # Create merge method
    try:
        method_class = {
            'slerp': SlerpMerge,
            'ties': TiesMerge, 
            'task_arithmetic': TaskArithmeticMerge,
            'linear': LinearMerge
        }[config.merge_method]
        
        # Convert to base merge config
        base_config = BaseMergeConfig(
            method=config.merge_method,
            models=config.get_model_list(),
            parameters=config.parameters.__dict__ if config.parameters else {},
            dtype=config.dtype,
            device=config.device,
            output_path=config.output_path,
            base_model=config.base_model
        )
        
        merge_method = method_class(base_config)
        print(f"🚀 Initialized {config.merge_method.upper()} merge method")
        
    except KeyError:
        print(f"❌ Unsupported merge method: {config.merge_method}")
        return 1
    except Exception as e:
        print(f"❌ Error initializing merge method: {e}")
        return 1
    
    # Load models
    try:
        print("📥 Loading models...")
        models = merge_method.load_models()
        print(f"✅ Loaded {len(models)} models")
        
        if args.verbose:
            for name, model in models.items():
                param_count = ModelUtils.count_parameters(model)
                print(f"  - {name}: {param_count:,} parameters")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return 1
    
    # Check memory pressure
    if memory_manager.check_memory_pressure():
        print("⚠️  High memory usage detected - consider using smaller models or CPU")
        if not args.allow_crimes:
            print("   Use --allow-crimes to proceed anyway")
            return 1
    
    # Perform merge
    try:
        print(f"🔀 Merging models using {config.merge_method.upper()}...")
        merged_model = merge_method.merge(models)
        print("✅ Models merged successfully")
        
        if args.verbose:
            merged_params = ModelUtils.count_parameters(merged_model)
            print(f"  Merged model: {merged_params:,} parameters")
        
    except Exception as e:
        print(f"❌ Error during merge: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    # Save merged model
    try:
        print(f"💾 Saving merged model to {config.output_path}...")
        
        # Create output directory
        os.makedirs(config.output_path, exist_ok=True)
        
        # Save model
        merge_method.save_merged_model(merged_model)
        
        # Save tokenizer if specified
        if config.tokenizer_source:
            from transformers import AutoTokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_source)
                tokenizer.save_pretrained(config.output_path)
                print(f"✅ Saved tokenizer from {config.tokenizer_source}")
            except Exception as e:
                print(f"⚠️  Warning: Could not save tokenizer: {e}")
        
        print(f"✅ Merge completed successfully! Output saved to {config.output_path}")
        
    except Exception as e:
        print(f"❌ Error saving merged model: {e}")
        return 1
    
    # Cleanup
    if args.cleanup:
        print("🧹 Cleaning up memory...")
        memory_manager.optimize_memory()
    
    return 0


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
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return 1
    
    return 0


def push_hf_command(args):
    """Push MoL model to Hugging Face Hub."""
    print(f"🚀 Pushing MoL model to Hugging Face: {args.repo_id}")
    
    # Setup logging
    setup_logging(args.log_level)
    
    from ..utils.hf_utils import HuggingFacePublisher, HF_HUB_AVAILABLE
    
    if not HF_HUB_AVAILABLE:
        print("❌ Hugging Face Hub not available. Install with:")
        print("   pip install huggingface_hub")
        return 1
    
    # Load MoL model
    try:
        if args.mol_checkpoint:
            # Load from MoL checkpoint
            from ..core.mol_runtime import MoLRuntime
            print(f"📥 Loading MoL checkpoint: {args.mol_checkpoint}")
            mol_runtime = MoLRuntime.load_checkpoint(args.mol_checkpoint)
        elif args.config:
            # Create from config and trained weights
            print(f"📥 Creating MoL runtime from config: {args.config}")
            parser = ConfigParser()
            config = parser.parse_config(args.config)
            
            # This would need additional implementation for loading trained MoL runtime
            print("❌ Loading from config not yet implemented. Use --mol-checkpoint instead.")
            return 1
        else:
            print("❌ Must specify either --mol-checkpoint or --config")
            return 1
            
    except Exception as e:
        print(f"❌ Error loading MoL model: {e}")
        return 1
    
    # Push to HuggingFace
    try:
        repo_url = mol_runtime.push_to_hf(
            repo_id=args.repo_id,
            fusion_type=args.fusion_type,
            token=args.token,
            commit_message=args.commit_message,
            private=args.private,
            create_pr=args.create_pr,
            fusion_method=args.fusion_method
        )
        
        print(f"✅ Successfully pushed to: {repo_url}")
        
        if args.fusion_type == "fused":
            print(f"📦 Created fully fused static model using {args.fusion_method}")
            print("   This model can be used as a standard HuggingFace model")
        else:
            print("🔄 Uploaded lightweight MoL runtime")
            print("   Requires MoL system to load and use")
        
    except Exception as e:
        print(f"❌ Error pushing to HuggingFace: {e}")
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


def examples_command(args):
    """Generate example configuration files."""
    print(f"📝 Generating example configurations in {args.output_dir}")
    
    parser = ConfigParser()
    try:
        parser.create_example_configs(args.output_dir)
        print("✅ Example configurations created successfully")
        print(f"   Check the {args.output_dir} directory for examples")
    except Exception as e:
        print(f"❌ Error creating examples: {e}")
        return 1
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="mol-merge",
        description="MoL (Modular Layer) model merging toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mol-merge config.yml ./merged_model
  mol-merge config.yml ./merged_model --device cuda --verbose
  mol-merge --validate config.yml
  mol-merge --examples ./config_examples
  mol-merge push-hf my-username/my-model --mol-checkpoint ./model.pt --fusion-type fused
        """
    )
    
    # Global arguments
    parser.add_argument(
        "--version",
        action="version",
        version="MoL Merge 0.2.0"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Merge command (default)
    merge_parser = subparsers.add_parser("merge", help="Merge models", aliases=["m"])
    merge_parser.add_argument(
        "config",
        help="Path to YAML configuration file"
    )
    merge_parser.add_argument(
        "output_path",
        nargs="?",
        help="Output path for merged model (overrides config)"
    )
    merge_parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        help="Device to use for merging (overrides config)"
    )
    merge_parser.add_argument(
        "--dtype",
        choices=["float16", "float32", "bfloat16"],
        help="Data type for merging (overrides config)"
    )
    merge_parser.add_argument(
        "--low-cpu-mem-usage",
        action="store_true",
        default=None,
        help="Use low CPU memory mode"
    )
    merge_parser.add_argument(
        "--no-low-cpu-mem-usage",
        dest="low_cpu_mem_usage",
        action="store_false",
        help="Disable low CPU memory mode"
    )
    merge_parser.add_argument(
        "--lazy-unpickle",
        action="store_true",
        default=None,
        help="Use lazy unpickling"
    )
    merge_parser.add_argument(
        "--no-lazy-unpickle",
        dest="lazy_unpickle",
        action="store_false",
        help="Disable lazy unpickling"
    )
    merge_parser.add_argument(
        "--allow-crimes",
        action="store_true",
        help="Allow potentially unsafe operations"
    )
    merge_parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up memory after merge"
    )
    merge_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    merge_parser.add_argument(
        "--use-safetensors",
        action="store_true",
        default=True,
        help="Use SafeTensors format for model serialization (default: True)"
    )
    merge_parser.add_argument(
        "--no-safetensors",
        dest="use_safetensors",
        action="store_false",
        help="Disable SafeTensors and use PyTorch format"
    )
    merge_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Allow remote code execution (SECURITY RISK: only use with trusted models)"
    )
    merge_parser.set_defaults(func=merge_command)
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate configuration", aliases=["v"])
    validate_parser.add_argument(
        "config",
        help="Path to YAML configuration file"
    )
    validate_parser.set_defaults(func=validate_command)
    
    # Examples command
    examples_parser = subparsers.add_parser("examples", help="Generate example configs", aliases=["e"])
    examples_parser.add_argument(
        "output_dir",
        nargs="?",
        default="./examples",
        help="Output directory for examples (default: ./examples)"
    )
    examples_parser.set_defaults(func=examples_command)
    
    # Push to HuggingFace command
    push_parser = subparsers.add_parser("push-hf", help="Push model to Hugging Face Hub", aliases=["push"])
    push_parser.add_argument(
        "repo_id",
        help="Repository ID (username/model-name)"
    )
    push_parser.add_argument(
        "--mol-checkpoint",
        help="Path to MoL checkpoint file"
    )
    push_parser.add_argument(
        "--config",
        help="Path to MoL configuration file (alternative to checkpoint)"
    )
    push_parser.add_argument(
        "--fusion-type",
        choices=["runtime", "fused"],
        default="runtime",
        help="Type of model to push (default: runtime)"
    )
    push_parser.add_argument(
        "--fusion-method",
        choices=["weighted_average", "best_expert", "learned_weights"],
        default="weighted_average",
        help="Fusion method for fused models (default: weighted_average)"
    )
    push_parser.add_argument(
        "--token",
        help="HuggingFace API token (or set HF_TOKEN env var)"
    )
    push_parser.add_argument(
        "--commit-message",
        help="Custom commit message"
    )
    push_parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repository"
    )
    push_parser.add_argument(
        "--create-pr",
        action="store_true",
        help="Create pull request instead of direct push"
    )
    push_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Allow remote code execution (SECURITY RISK: only use with trusted models)"
    )
    push_parser.set_defaults(func=push_hf_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle default command (merge)
    if not hasattr(args, 'func'):
        if len(sys.argv) >= 2 and not sys.argv[1].startswith('-'):
            # Treat as merge command
            args.command = "merge"
            args.config = sys.argv[1]
            args.output_path = sys.argv[2] if len(sys.argv) >= 3 else None
            args.device = None
            args.dtype = None
            args.low_cpu_mem_usage = None
            args.lazy_unpickle = None
            args.allow_crimes = False
            args.cleanup = False
            args.verbose = False
            args.trust_remote_code = False
            args.func = merge_command
        else:
            parser.print_help()
            return 1
    
    # Execute command
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())