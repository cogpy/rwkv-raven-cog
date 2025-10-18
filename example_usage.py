#!/usr/bin/env python3
"""
Example usage of the RWKV-Raven OpenCog Transformer

This script demonstrates how to use the OpenCog transformation
for RWKV models in various scenarios.
"""

import json
import os
from opencog_transform import OpenCogTransformer, OpenCogConfig

def example_basic_transformation():
    """Basic example of transforming a single model"""
    print("=== Basic Transformation Example ===")
    
    # Create configuration
    config = OpenCogConfig(
        atomspace_size=10000,
        attention_threshold=0.5,
        pattern_match_depth=3
    )
    
    # Initialize transformer
    transformer = OpenCogTransformer(config)
    
    # Transform model (will use mock data if actual model not available)
    model_path = 'rwkv-4-raven/RWKV-4-Raven-3B-v12-Eng98%25-Other2%25-20230520-ctx4096.pth'
    output_dir = 'example_output/basic_transform'
    
    result = transformer.transform_model(model_path, output_dir)
    
    print(f"Transformation completed!")
    print(f"Generated {len(result['cognitive_layers'])} cognitive layers")
    print(f"Created {len(result['atomspace_nodes'])} Atomspace nodes")
    print(f"Configuration saved to: {result['config_path']}")
    
    return result

def example_custom_configuration():
    """Example with custom cognitive modules and settings"""
    print("\n=== Custom Configuration Example ===")
    
    # Create custom configuration
    config = OpenCogConfig(
        atomspace_size=20000,
        attention_threshold=0.7,
        pattern_match_depth=5,
        cognitive_modules=[
            'perception', 'attention', 'memory', 
            'reasoning', 'language', 'emotion'
        ],
        enable_symbolic_integration=True,
        memory_optimization=False
    )
    
    transformer = OpenCogTransformer(config)
    
    # Transform with custom settings
    model_path = 'rwkv-4-raven/RWKV-4-Raven-1B5-v12-Eng98%25-Other2%25-20230520-ctx4096.pth'
    output_dir = 'example_output/custom_transform'
    
    result = transformer.transform_model(model_path, output_dir)
    
    print(f"Custom transformation completed with {len(config.cognitive_modules)} modules")
    return result

def example_analyze_transformation():
    """Example of analyzing transformation results"""
    print("\n=== Transformation Analysis Example ===")
    
    # Load a transformation configuration
    config_path = 'opencog_transformed/RWKV-4-Raven-3B-v12-Eng98%25-Other2%25-20230520-ctx4096/opencog_config.json'
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        print("Transformation Analysis:")
        print("-" * 30)
        
        # Analyze cognitive modules
        layer_mapping = config_data['layer_mapping']
        module_counts = {}
        attention_stats = []
        
        for layer_name, layer_info in layer_mapping.items():
            module_type = layer_info['module_type']
            module_counts[module_type] = module_counts.get(module_type, 0) + 1
            attention_stats.append(layer_info['attention_value'])
        
        print(f"Total layers: {len(layer_mapping)}")
        print("Module distribution:")
        for module, count in module_counts.items():
            print(f"  {module}: {count} layers")
        
        print(f"Average attention value: {sum(attention_stats) / len(attention_stats):.3f}")
        print(f"Max attention value: {max(attention_stats):.3f}")
        print(f"Min attention value: {min(attention_stats):.3f}")
        
        # Analyze patterns
        pattern_types = set()
        for layer_info in layer_mapping.values():
            for pattern in layer_info['pattern_templates']:
                if 'AttentionLink' in pattern:
                    pattern_types.add('Attention')
                elif 'InferenceLink' in pattern:
                    pattern_types.add('Inference')
                elif 'EvaluationLink' in pattern:
                    pattern_types.add('Evaluation')
        
        print(f"Pattern types found: {', '.join(pattern_types)}")
    else:
        print(f"Configuration file not found: {config_path}")
        print("Please run a transformation first.")

def example_cognitive_layer_inspection():
    """Example of inspecting individual cognitive layers"""
    print("\n=== Cognitive Layer Inspection Example ===")
    
    config = OpenCogConfig()
    transformer = OpenCogTransformer(config)
    
    # Create a simple mock model for demonstration
    mock_model = transformer._create_mock_model_structure()
    
    # Transform layers
    cognitive_layers = transformer.transform_layers(mock_model)
    
    print("Inspecting cognitive layers:")
    print("-" * 40)
    
    for layer_name, layer in cognitive_layers.items():
        print(f"\nLayer: {layer_name}")
        print(f"  Module Type: {layer.module_type}")
        print(f"  Input Size: {layer.input_size}")
        print(f"  Output Size: {layer.output_size}")
        print(f"  Pattern Templates: {len(layer.pattern_templates)}")
        
        for i, pattern in enumerate(layer.pattern_templates):
            print(f"    Pattern {i+1}: {pattern}")

def example_attention_network():
    """Example of creating and analyzing attention networks"""
    print("\n=== Attention Network Example ===")
    
    config = OpenCogConfig(
        attention_threshold=0.6
    )
    transformer = OpenCogTransformer(config)
    
    # Create mock layers
    mock_model = transformer._create_mock_model_structure()
    cognitive_layers = transformer.transform_layers(mock_model)
    
    # Create attention network
    attention_map = transformer.create_attention_network(cognitive_layers)
    
    print("Attention Network Analysis:")
    print("-" * 30)
    
    # Sort by attention value
    sorted_attention = sorted(attention_map.items(), 
                            key=lambda x: x[1], reverse=True)
    
    print("Top attention nodes:")
    for i, (layer_name, attention_value) in enumerate(sorted_attention[:10]):
        print(f"  {i+1}. {layer_name}: {attention_value:.2f}")
    
    # Find layers above threshold
    high_attention = [name for name, value in attention_map.items() 
                     if value >= config.attention_threshold]
    
    print(f"\nLayers above attention threshold ({config.attention_threshold}): {len(high_attention)}")
    print("High attention layers:")
    for layer_name in high_attention:
        print(f"  - {layer_name}")

def main():
    """Run all examples"""
    print("RWKV-Raven OpenCog Transformer Examples")
    print("=" * 50)
    
    # Create example output directory
    os.makedirs('example_output', exist_ok=True)
    
    try:
        # Run examples
        example_basic_transformation()
        example_custom_configuration()
        example_analyze_transformation()
        example_cognitive_layer_inspection()
        example_attention_network()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("\nGenerated files:")
        for root, dirs, files in os.walk('example_output'):
            for file in files:
                filepath = os.path.join(root, file)
                print(f"  {filepath}")
                
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()