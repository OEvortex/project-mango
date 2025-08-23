"""
CLI utilities for generating example configurations.
"""

import os
from ..config import ConfigParser


def examples_command(args):
    """Generate example configuration files."""
    print(f"📝 Generating example configurations in {args.output_dir}")
    
    parser = ConfigParser()
    try:
        parser.create_example_configs(args.output_dir)
        print("✅ Example configurations created successfully")
        print(f"   Check the {args.output_dir} directory for examples")
        
        # List created files
        if os.path.exists(args.output_dir):
            files = [f for f in os.listdir(args.output_dir) if f.endswith('.yml')]
            if files:
                print(f"\n📄 Created example files:")
                for file in sorted(files):
                    print(f"  - {file}")
                    
                print(f"\n💡 Try running:")
                print(f"   mol-merge validate {args.output_dir}/slerp_example.yml")
                print(f"   mol-merge {args.output_dir}/slerp_example.yml ./my_merged_model")
        
    except Exception as e:
        print(f"❌ Error creating examples: {e}")
        return 1
    
    return 0


def create_custom_config(
    merge_method: str,
    models: list,
    output_path: str,
    config_file: str,
    **kwargs
):
    """Create a custom configuration file."""
    print(f"📝 Creating {merge_method} configuration for {len(models)} models")
    
    # Build configuration dictionary
    config_dict = {
        'merge_method': merge_method,
        'slices': [{'model': model} for model in models],
        'dtype': kwargs.get('dtype', 'float16'),
        'output_path': output_path,
        'device': kwargs.get('device', 'cpu')
    }
    
    # Add method-specific parameters
    if merge_method == 'slerp':
        config_dict['parameters'] = {
            't': kwargs.get('t', 0.5),
            'eps': kwargs.get('eps', 1e-8)
        }
    elif merge_method == 'ties':
        config_dict['base_model'] = models[0]
        config_dict['parameters'] = {
            'density': kwargs.get('density', 0.8),
            'normalize': kwargs.get('normalize', True),
            'majority_sign_method': kwargs.get('majority_sign_method', 'total')
        }
    elif merge_method == 'task_arithmetic':
        config_dict['base_model'] = models[0]
        weights = kwargs.get('weights', {})
        if not weights:
            weights = {model: 1.0 for model in models[1:]}
        config_dict['parameters'] = {
            'weights': weights,
            'scaling_factor': kwargs.get('scaling_factor', 1.0),
            'normalize': kwargs.get('normalize', False)
        }
    elif merge_method == 'linear':
        weights = kwargs.get('weights', {})
        if not weights:
            weights = {model: 1.0/len(models) for model in models}
        config_dict['parameters'] = {
            'weights': weights,
            'normalize_weights': kwargs.get('normalize_weights', True)
        }
    
    # Save configuration
    try:
        import yaml
        
        # Create directory if needed
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        print(f"✅ Created configuration file: {config_file}")
        
        # Validate the created config
        parser = ConfigParser()
        if parser.validate_file_syntax(config_file):
            print("✅ Configuration syntax is valid")
        else:
            print("⚠️  Warning: Configuration syntax may have issues")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating configuration: {e}")
        return False


def list_merge_methods():
    """List available merge methods with descriptions."""
    methods = {
        'slerp': {
            'name': 'Spherical Linear Interpolation',
            'description': 'Smoothly interpolates between model weights in spherical space',
            'best_for': 'Merging 2 similar models, preserving geometric properties',
            'parameters': ['t (interpolation factor)', 'eps (numerical stability)']
        },
        'ties': {
            'name': 'TIES (Trim, Elect Sign, Disjoint Merge)',
            'description': 'Resolves parameter conflicts when merging multiple task-specific models',
            'best_for': 'Merging multiple specialized models, handling parameter disagreements',
            'parameters': ['density (parameter retention)', 'normalize', 'majority_sign_method']
        },
        'task_arithmetic': {
            'name': 'Task Arithmetic',
            'description': 'Combines task vectors (difference from base model)',
            'best_for': 'Adding/removing capabilities, combining task-specific fine-tunes',
            'parameters': ['weights (per model)', 'scaling_factor', 'normalize']
        },
        'linear': {
            'name': 'Linear Interpolation',
            'description': 'Weighted average of model parameters',
            'best_for': 'Simple blending, when models are very similar',
            'parameters': ['weights (per model)', 'normalize_weights']
        }
    }
    
    print("🔧 Available Merge Methods:\n")
    
    for method_id, info in methods.items():
        print(f"📌 {method_id.upper()}: {info['name']}")
        print(f"   {info['description']}")
        print(f"   🎯 Best for: {info['best_for']}")
        print(f"   ⚙️  Parameters: {', '.join(info['parameters'])}")
        print()


def create_interactive_config():
    """Create configuration interactively."""
    print("🔧 Interactive Configuration Creator")
    print("This will help you create a merge configuration step by step.\n")
    
    # Choose merge method
    print("1️⃣  Choose merge method:")
    list_merge_methods()
    
    while True:
        method = input("Enter merge method (slerp/ties/task_arithmetic/linear): ").strip().lower()
        if method in ['slerp', 'ties', 'task_arithmetic', 'linear']:
            break
        print("❌ Invalid method. Please choose from: slerp, ties, task_arithmetic, linear")
    
    # Get models
    print(f"\n2️⃣  Enter models to merge (one per line, empty line to finish):")
    models = []
    while True:
        model = input(f"Model {len(models)+1}: ").strip()
        if not model:
            break
        models.append(model)
    
    if len(models) < 2:
        print("❌ Need at least 2 models to merge")
        return False
    
    # Get output path
    output_path = input("\n3️⃣  Output path for merged model: ").strip()
    if not output_path:
        output_path = "./merged_model"
    
    # Get config file path
    config_file = input("4️⃣  Configuration file to create: ").strip()
    if not config_file:
        config_file = f"{method}_config.yml"
    
    # Method-specific parameters
    kwargs = {}
    
    if method == 'slerp':
        t = input("5️⃣  Interpolation factor t (0-1, default 0.5): ").strip()
        if t:
            try:
                kwargs['t'] = float(t)
            except ValueError:
                print("⚠️  Invalid t value, using default 0.5")
    
    elif method == 'ties':
        density = input("5️⃣  Density (fraction of parameters to keep, default 0.8): ").strip()
        if density:
            try:
                kwargs['density'] = float(density)
            except ValueError:
                print("⚠️  Invalid density, using default 0.8")
    
    elif method == 'task_arithmetic':
        print("5️⃣  Model weights (press enter for equal weights):")
        weights = {}
        for model in models[1:]:  # Skip base model
            weight = input(f"   Weight for {model} (default 1.0): ").strip()
            if weight:
                try:
                    weights[model] = float(weight)
                except ValueError:
                    weights[model] = 1.0
            else:
                weights[model] = 1.0
        if weights:
            kwargs['weights'] = weights
    
    elif method == 'linear':
        print("5️⃣  Model weights (press enter for equal weights):")
        weights = {}
        for model in models:
            weight = input(f"   Weight for {model} (default {1.0/len(models):.2f}): ").strip()
            if weight:
                try:
                    weights[model] = float(weight)
                except ValueError:
                    weights[model] = 1.0/len(models)
            else:
                weights[model] = 1.0/len(models)
        if weights:
            kwargs['weights'] = weights
    
    # Device and dtype
    device = input("\n6️⃣  Device (cpu/cuda, default cpu): ").strip()
    if device:
        kwargs['device'] = device
    
    dtype = input("7️⃣  Data type (float16/float32/bfloat16, default float16): ").strip()
    if dtype:
        kwargs['dtype'] = dtype
    
    # Create configuration
    print(f"\n🚀 Creating {method} configuration...")
    success = create_custom_config(method, models, output_path, config_file, **kwargs)
    
    if success:
        print(f"\n✅ Configuration created successfully!")
        print(f"📄 File: {config_file}")
        print(f"\n💡 Next steps:")
        print(f"   1. Validate: mol-merge validate {config_file}")
        print(f"   2. Run merge: mol-merge {config_file}")
    
    return success