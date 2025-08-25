#!/usr/bin/env python3
"""
MoL Model Inference Script (check.py)

This script loads a trained MoL model and provides flexible inference capabilities
including batch processing, interactive mode, and routing analysis.

Usage:
    python check.py [--model MODEL_PATH] [--interactive] [--batch] [--analyze-routing]
    
Examples:
    # Interactive mode
    python check.py --model ./mol_trained_model.safetensors --interactive
    
    # Batch inference 
    python check.py --model ./mol_trained_model.safetensors --batch
    
    # Analyze routing behavior
    python check.py --model ./mol_trained_model.safetensors --analyze-routing
"""

import argparse
import torch
import logging
from pathlib import Path
from typing import List, Dict, Optional
import json
from datetime import datetime

# Import MoL components
from mol.core.mol_runtime import MoLRuntime
from mol.utils.model_utils import ModelUtils

# Setup logging with Unicode support
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Fix for Windows Unicode handling - only redirect if needed
import sys
try:
    # Test if stdout works properly
    sys.stdout.write("")
    sys.stderr.write("")
except (UnicodeEncodeError, ValueError):
    # Only redirect if there's an encoding issue
    import codecs
    if sys.platform == "win32":
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())


class MoLInferenceEngine:
    """Inference engine for trained MoL models."""
    
    def __init__(self, model_path: str):
        """Initialize the inference engine."""
        self.model_path = Path(model_path)
        self.mol_runtime = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🚀 MoL Inference Engine")
        print(f"📁 Model path: {self.model_path}")
        print(f"🖥️  Device: {self.device}")
        print("=" * 60)
        
        self.load_model()
    
    def load_model(self):
        """Load the trained MoL model."""
        try:
            print("📥 Loading MoL model...")
            
            # Check if file exists
            if not self.model_path.exists():
                # Try different extensions
                for ext in ['.safetensors', '.pt', '']:
                    test_path = Path(str(self.model_path) + ext)
                    if test_path.exists():
                        self.model_path = test_path
                        break
                else:
                    raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Load using MoLRuntime's load_checkpoint method
            self.mol_runtime = MoLRuntime.load_checkpoint(str(self.model_path))
            self.mol_runtime.to(self.device)
            self.mol_runtime.eval()
            
            print(f"✅ Model loaded successfully")
            self.print_model_info()
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def print_model_info(self):
        """Print detailed information about the loaded model."""
        print("\\n📋 Model Information:")
        
        # Model configuration
        config = self.mol_runtime.config
        print(f"   • Expert models: {config.models}")
        print(f"   • Adapter type: {config.adapter_type}")
        print(f"   • Router type: {config.router_type}")
        print(f"   • Number of layers: {len(self.mol_runtime.layers)}")
        print(f"   • Target hidden dim: {self.mol_runtime.target_hidden_dim}")
        
        # Parameter counts
        total_params = ModelUtils.count_parameters(self.mol_runtime)
        trainable_params = ModelUtils.count_parameters(self.mol_runtime, trainable_only=True)
        
        print(f"   • Total parameters: {total_params:,}")
        print(f"   • Trainable parameters: {trainable_params:,}")
        print(f"   • Trainable ratio: {trainable_params/total_params*100:.2f}%")
        
        # Memory usage
        memory_stats = self.mol_runtime.memory_manager.get_memory_stats()
        print(f"   • RAM usage: {memory_stats.used_ram:.1f}GB / {memory_stats.total_ram:.1f}GB")
        if memory_stats.total_vram > 0:
            print(f"   • VRAM usage: {memory_stats.used_vram:.1f}GB / {memory_stats.total_vram:.1f}GB")
    
    def generate_text(
        self, 
        prompt: str, 
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        do_sample: bool = True,
        top_p: float = 0.9,
        top_k: int = 50,
        return_routing_stats: bool = False
    ) -> Dict:
        """Generate text from a prompt with optional routing analysis."""
        try:
            # Tokenize input
            inputs = self.mol_runtime.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            input_length = inputs['input_ids'].shape[1]
            
            with torch.no_grad():
                # Generate text
                generated = self.mol_runtime.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                    top_k=top_k,
                    pad_token_id=self.mol_runtime.tokenizer.eos_token_id
                )
                
                # Decode generated text
                full_text = self.mol_runtime.tokenizer.decode(generated[0], skip_special_tokens=True)
                generated_text = self.mol_runtime.tokenizer.decode(
                    generated[0][input_length:], skip_special_tokens=True
                )
                
                result = {
                    'prompt': prompt,
                    'generated': generated_text,
                    'full_text': full_text,
                    'input_tokens': input_length,
                    'output_tokens': len(generated[0]) - input_length
                }
                
                # Add routing analysis if requested
                if return_routing_stats:
                    _, router_stats = self.mol_runtime.forward(
                        inputs['input_ids'],
                        inputs['attention_mask'],
                        return_router_stats=True
                    )
                    result['routing_stats'] = router_stats
                
                return result
                
        except Exception as e:
            return {
                'prompt': prompt,
                'error': str(e),
                'generated': '',
                'full_text': prompt
            }
    
    def analyze_routing(self, texts: List[str]) -> Dict:
        """Analyze routing behavior across multiple inputs."""
        print("\\n🧭 Routing Analysis")
        print("=" * 40)
        
        routing_analysis = {
            'inputs': [],
            'summary': {},
            'timestamp': datetime.now().isoformat()
        }
        
        with torch.no_grad():
            for i, text in enumerate(texts):
                print(f"\\nAnalyzing input {i+1}: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                
                # Tokenize
                inputs = self.mol_runtime.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128
                ).to(self.device)
                
                # Forward pass with routing stats
                _, router_stats = self.mol_runtime.forward(
                    inputs['input_ids'],
                    inputs['attention_mask'],
                    return_router_stats=True
                )
                
                if router_stats:
                    text_analysis = {
                        'text': text,
                        'token_count': inputs['input_ids'].shape[1],
                        'layers': {}
                    }
                    
                    for layer_name, stats in router_stats.items():
                        expert_weights = stats['expert_weights']  # [batch, seq, experts]
                        avg_weights = expert_weights.mean(dim=(0, 1))  # Average per expert
                        
                        layer_analysis = {
                            'router_entropy': float(stats['router_entropy']),
                            'load_balancing_loss': float(stats['load_balancing_loss']),
                            'expert_usage': avg_weights.tolist(),
                            'dominant_expert': int(avg_weights.argmax())
                        }
                        
                        text_analysis['layers'][layer_name] = layer_analysis
                        
                        print(f"   {layer_name}:")
                        print(f"     Entropy: {layer_analysis['router_entropy']:.3f}")
                        print(f"     Dominant expert: {layer_analysis['dominant_expert']}")
                        print(f"     Usage: {[f'{w:.3f}' for w in layer_analysis['expert_usage']]}")
                    
                    routing_analysis['inputs'].append(text_analysis)
        
        # Compute summary statistics
        if routing_analysis['inputs']:
            layer_names = list(routing_analysis['inputs'][0]['layers'].keys())
            summary = {}
            
            for layer_name in layer_names:
                entropies = [inp['layers'][layer_name]['router_entropy'] for inp in routing_analysis['inputs']]
                expert_usages = [inp['layers'][layer_name]['expert_usage'] for inp in routing_analysis['inputs']]
                
                summary[layer_name] = {
                    'avg_entropy': sum(entropies) / len(entropies),
                    'entropy_std': torch.tensor(entropies).std().item(),
                    'avg_expert_usage': torch.tensor(expert_usages).mean(dim=0).tolist()
                }
            
            routing_analysis['summary'] = summary
            
            print("\\n📊 Summary Statistics:")
            for layer_name, stats in summary.items():
                print(f"   {layer_name}:")
                print(f"     Avg entropy: {stats['avg_entropy']:.3f} ± {stats['entropy_std']:.3f}")
                print(f"     Avg expert usage: {[f'{u:.3f}' for u in stats['avg_expert_usage']]}")
        
        return routing_analysis
    
    def interactive_mode(self):
        """Run interactive inference mode."""
        print("\\n🎭 Interactive Mode")
        print("=" * 40)
        print("Enter your prompts (type 'quit' to exit, 'analyze' for routing analysis)")
        print("Options: --tokens N (set max tokens), --temp N (set temperature)")
        
        max_tokens = 50
        temperature = 0.8
        
        while True:
            try:
                user_input = input("\\n> ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'analyze':
                    print("Enter texts to analyze (one per line, empty line to finish):")
                    texts = []
                    while True:
                        text = input("  - ").strip()
                        if not text:
                            break
                        texts.append(text)
                    
                    if texts:
                        self.analyze_routing(texts)
                    continue
                
                # Parse options
                if user_input.startswith('--'):
                    parts = user_input.split()
                    if len(parts) >= 2:
                        if parts[0] == '--tokens':
                            max_tokens = int(parts[1])
                            print(f"Set max tokens to {max_tokens}")
                        elif parts[0] == '--temp':
                            temperature = float(parts[1])
                            print(f"Set temperature to {temperature}")
                    continue
                
                if not user_input:
                    continue
                
                # Generate response
                print("🤔 Generating...")
                result = self.generate_text(
                    user_input,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    return_routing_stats=True
                )
                
                if 'error' in result:
                    print(f"❌ Error: {result['error']}")
                else:
                    print(f"✨ Generated ({result['output_tokens']} tokens):")
                    print(f"   {result['generated']}")
                    
                    # Show dominant experts
                    if 'routing_stats' in result:
                        print("\\n🧭 Routing info:")
                        for layer_name, stats in result['routing_stats'].items():
                            expert_weights = stats['expert_weights'].mean(dim=(0, 1))
                            dominant = expert_weights.argmax().item()
                            print(f"   {layer_name}: Expert {dominant} ({expert_weights[dominant]:.3f})")
            
            except KeyboardInterrupt:
                print("\\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def batch_mode(self, prompts: Optional[List[str]] = None):
        """Run batch inference on multiple prompts."""
        print("\\n📦 Batch Mode")
        print("=" * 40)
        
        if prompts is None:
            # Default test prompts
            prompts = [
                "Hello, how are you today?",
                "Explain artificial intelligence in simple terms.",
                "Once upon a time in a magical forest,",
                "The benefits of renewable energy include",
                "My favorite programming language is Python because",
                "In a world where technology advances rapidly,",
                "The key to successful machine learning is",
                "Climate change is an important issue because"
            ]
        
        print(f"Processing {len(prompts)} prompts...")
        
        results = []
        for i, prompt in enumerate(prompts):
            print(f"\\n[{i+1}/{len(prompts)}] Processing: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
            
            result = self.generate_text(prompt, max_new_tokens=30, return_routing_stats=True)
            results.append(result)
            
            if 'error' not in result:
                print(f"   ✅ Generated: '{result['generated'][:100]}{'...' if len(result['generated']) > 100 else ''}'")
            else:
                print(f"   ❌ Error: {result['error']}")
        
        # Save results
        output_file = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\\n✅ Batch processing complete. Results saved to {output_file}")
        return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="MoL Model Inference Script")
    parser.add_argument("--model", "-m", default="./mol_trained_model.safetensors",
                        help="Path to trained MoL model")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--batch", "-b", action="store_true",
                        help="Run batch inference")
    parser.add_argument("--analyze-routing", "-a", action="store_true",
                        help="Analyze routing behavior")
    parser.add_argument("--prompts", nargs="+",
                        help="Custom prompts for batch mode")
    
    args = parser.parse_args()
    
    try:
        # Initialize inference engine
        engine = MoLInferenceEngine(args.model)
        
        # Run requested mode
        if args.interactive:
            engine.interactive_mode()
        elif args.batch:
            engine.batch_mode(args.prompts)
        elif args.analyze_routing:
            # Default analysis texts
            analysis_texts = [
                "Hello, how are you?",
                "Explain quantum physics",
                "Write a story about",
                "The economic situation",
                "My programming experience"
            ]
            engine.analyze_routing(analysis_texts)
        else:
            # Default: quick demo
            print("\\n🎯 Quick Demo")
            print("=" * 40)
            
            demo_prompts = [
                "Hello, how are you today?",
                "Explain machine learning briefly.",
                "Once upon a time,"
            ]
            
            for prompt in demo_prompts:
                print(f"\\nPrompt: '{prompt}'")
                result = engine.generate_text(prompt, max_new_tokens=20)
                if 'error' not in result:
                    print(f"Output: '{result['generated']}'")
                else:
                    print(f"Error: {result['error']}")
            
            print("\\n💡 Use --interactive for interactive mode, --batch for batch processing")
    
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())