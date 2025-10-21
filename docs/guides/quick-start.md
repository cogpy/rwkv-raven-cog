# Quick Start Guide

Get up and running with RWKV-Raven-Cog in just a few minutes! This guide will walk you through installation, basic usage, and your first transformation.

## 🚀 Quick Installation

### Prerequisites

Before you begin, ensure you have:
- Python 3.8 or higher
- Git LFS (for large model files)
- At least 4GB of RAM (8GB+ recommended for large models)

### Option 1: One-Line Setup (Recommended)

```bash
# Clone and setup everything automatically
curl -sSL https://raw.githubusercontent.com/cogpy/rwkv-raven-cog/main/setup_environment.sh | bash
```

### Option 2: Manual Installation

```bash
# Clone the repository
git clone https://github.com/cogpy/rwkv-raven-cog.git
cd rwkv-raven-cog

# Install the package
pip install -e .

# Or install from PyPI (when available)
pip install rwkv-raven-cog
```

### Verify Installation

```bash
# Test the command line interface
rwkv-opencog-transform --help

# Test Python import
python -c "from opencog_transform import OpenCogTransformer; print('✓ Installation successful!')"
```

## 🎯 Your First Transformation

### Step 1: Basic Python Usage

Create a simple Python script to transform a model:

```python
from opencog_transform import OpenCogTransformer, OpenCogConfig

# Create configuration
config = OpenCogConfig(
    atomspace_size=15000,
    attention_threshold=0.6,
    pattern_match_depth=4
)

# Initialize transformer
transformer = OpenCogTransformer(config)

# Transform a model (uses mock data if model file not available)
result = transformer.transform_model(
    model_path='rwkv-4-raven/RWKV-4-Raven-3B-v12-Eng98%25-Other2%25-20230520-ctx4096.pth',
    output_dir='my_first_transformation'
)

print(f"✓ Created {len(result['cognitive_layers'])} cognitive layers")
print(f"✓ Generated {len(result['atomspace_nodes'])} symbolic nodes")
print(f"✓ Configuration saved to: {result['config_path']}")
```

### Step 2: Command Line Usage

Transform models directly from the command line:

```bash
# Basic transformation
python opencog_transform.py

# Or using the installed command
rwkv-opencog-transform
```

### Step 3: Explore the Results

After transformation, you'll find these files in your output directory:

```
my_first_transformation/
├── opencog_config.json      # Complete transformation configuration
├── cognitive_layers.json    # Detailed layer information  
├── atomspace_nodes.json     # Symbolic knowledge representation
└── transformation_log.txt   # Processing log and metrics
```

## 📊 Understanding the Output

### Configuration File Structure

```json
{
  "opencog_config": {
    "atomspace_size": 15000,
    "attention_threshold": 0.6,
    "cognitive_modules": ["perception", "attention", "reasoning", "language"]
  },
  "layer_mapping": {
    "embedding": {
      "module_type": "perception",
      "input_size": 50277,
      "output_size": 2048,
      "attention_value": 0.8
    }
  },
  "transformation_metadata": {
    "total_layers": 50,
    "processing_time": 12.5,
    "memory_usage_mb": 256
  }
}
```

### Cognitive Layer Types

Your transformation will create several types of cognitive layers:

| Layer Type | Function | Example Components |
|-----------|----------|-------------------|
| **Perception** | Input processing | Word embeddings, feature extraction |
| **Attention** | Focus management | Attention mechanisms, salience detection |
| **Reasoning** | Logical inference | Feed-forward networks, pattern matching |
| **Language** | Communication | Output generation, token prediction |
| **Memory** | Knowledge storage | Layer normalization, residual connections |

## 🎮 Interactive Examples

### Example 1: Custom Configuration

```python
from opencog_transform import OpenCogTransformer, OpenCogConfig

# Create a research-oriented configuration
research_config = OpenCogConfig(
    atomspace_size=50000,
    attention_threshold=0.7,
    pattern_match_depth=6,
    cognitive_modules=[
        'perception', 'attention', 'reasoning', 
        'language', 'memory', 'emotion'
    ],
    enable_symbolic_integration=True,
    memory_optimization=False  # Disable for maximum detail
)

transformer = OpenCogTransformer(research_config)
result = transformer.transform_model('model.pth', 'research_output')
```

### Example 2: Performance-Optimized Setup

```python
# Configuration optimized for speed and memory efficiency
performance_config = OpenCogConfig(
    atomspace_size=10000,
    attention_threshold=0.5,
    pattern_match_depth=3,
    memory_optimization=True,
    cognitive_modules=['perception', 'attention', 'reasoning', 'language']
)

transformer = OpenCogTransformer(performance_config)
result = transformer.transform_model('model.pth', 'fast_output')
```

### Example 3: Analyzing Results

```python
import json

# Load and analyze transformation results
with open('my_first_transformation/opencog_config.json', 'r') as f:
    config_data = json.load(f)

# Analyze cognitive modules
layer_mapping = config_data['layer_mapping']
module_counts = {}

for layer_name, layer_info in layer_mapping.items():
    module_type = layer_info['module_type']
    module_counts[module_type] = module_counts.get(module_type, 0) + 1

print("Cognitive Module Distribution:")
for module, count in module_counts.items():
    print(f"  {module.title()}: {count} layers")

# Find highest attention layers
attention_layers = [(name, info['attention_value']) 
                   for name, info in layer_mapping.items()]
attention_layers.sort(key=lambda x: x[1], reverse=True)

print(f"\nTop 3 Attention Layers:")
for name, attention in attention_layers[:3]:
    print(f"  {name}: {attention:.3f}")
```

## 🔧 Customization Options

### Available Cognitive Modules

You can customize which cognitive modules to include:

```python
# Minimal setup (fastest)
minimal_modules = ['perception', 'attention', 'reasoning', 'language']

# Standard setup (balanced)
standard_modules = ['perception', 'attention', 'reasoning', 'language', 'memory']

# Complete setup (most detailed)  
complete_modules = [
    'perception', 'attention', 'reasoning', 'language', 
    'memory', 'emotion', 'action', 'learning'
]

config = OpenCogConfig(cognitive_modules=complete_modules)
```

### Performance Tuning

```python
# For large models (7B+ parameters)
large_model_config = OpenCogConfig(
    atomspace_size=25000,
    attention_threshold=0.7,
    memory_optimization=True,
    pattern_match_depth=4
)

# For small models (1.5B parameters)
small_model_config = OpenCogConfig(
    atomspace_size=8000,
    attention_threshold=0.5,
    memory_optimization=False,
    pattern_match_depth=5
)
```

## 🔍 Debugging and Troubleshooting

### Common Issues and Solutions

#### Issue: "Model file not found"
```python
# The system will automatically use mock data for testing
# To use real models, ensure the model file path is correct:

import os
model_path = 'rwkv-4-raven/model.pth'
if os.path.exists(model_path):
    print("✓ Model file found")
else:
    print("⚠ Using mock model data for demonstration")
```

#### Issue: "Out of memory"
```python
# Reduce configuration parameters:
memory_efficient_config = OpenCogConfig(
    atomspace_size=5000,        # Smaller knowledge base
    attention_threshold=0.8,    # Higher threshold filters more
    memory_optimization=True,   # Enable all optimizations
    pattern_match_depth=2       # Simpler reasoning
)
```

#### Issue: "Slow processing"
```python
# Optimize for speed:
fast_config = OpenCogConfig(
    atomspace_size=8000,
    cognitive_modules=['perception', 'attention', 'language'],  # Fewer modules
    pattern_match_depth=3,
    memory_optimization=True
)
```

### Logging and Diagnostics

```python
import logging

# Enable detailed logging
logging.basicConfig(level=logging.INFO)

# Transform with logging
transformer = OpenCogTransformer(config)
result = transformer.transform_model('model.pth', 'output', verbose=True)

# Check system diagnostics
from opencog_transform import get_system_diagnostics
diagnostics = get_system_diagnostics()
print(f"Available memory: {diagnostics['available_memory_gb']:.1f} GB")
print(f"CPU cores: {diagnostics['cpu_count']}")
```

## 📈 Next Steps

Now that you have RWKV-Raven-Cog working, explore these advanced topics:

### 1. Advanced Configuration
- Read the [Configuration Guide](../api/configuration.md) for detailed parameter descriptions
- Learn about [Performance Optimization](../guides/performance.md)

### 2. Integration with OpenCog
- Explore [Atomspace Integration](../architecture/atomspace-integration.md)
- Learn about [Pattern Matching](../guides/pattern-matching.md)

### 3. Custom Extensions
- Check the [Extension Points](../api/extensions.md) for customization options
- Read about [Custom Cognitive Modules](../guides/custom-modules.md)

### 4. Production Deployment
- Review the [Deployment Guide](../guides/deployment.md)
- Learn about [Monitoring and Maintenance](../guides/monitoring.md)

## 💡 Tips for Success

### Best Practices

1. **Start Small**: Begin with smaller models and basic configurations
2. **Monitor Resources**: Keep an eye on memory usage, especially with large models
3. **Validate Results**: Always check the output files for expected structure
4. **Incremental Complexity**: Gradually increase configuration complexity
5. **Regular Updates**: Keep the software updated for latest improvements

### Performance Tips

```python
# Optimal configuration for most use cases
optimal_config = OpenCogConfig(
    atomspace_size=15000,       # Good balance of detail and performance
    attention_threshold=0.6,    # Moderate selectivity
    pattern_match_depth=4,      # Reasonable reasoning depth
    memory_optimization=True,   # Always enable for production
    cognitive_modules=[         # Core modules for most applications
        'perception', 'attention', 'reasoning', 'language'
    ]
)
```

### Resource Planning

| Model Size | Recommended RAM | Atomspace Size | Processing Time |
|-----------|-----------------|----------------|-----------------|
| 1.5B params | 4GB | 8,000-12,000 | 2-5 minutes |
| 3B params | 6GB | 12,000-18,000 | 5-10 minutes |
| 7B params | 12GB | 18,000-25,000 | 10-20 minutes |

## 🤝 Getting Help

If you encounter issues:

1. **Check the logs** in your output directory
2. **Review the [Troubleshooting Guide](../guides/troubleshooting.md)**
3. **Search existing [GitHub Issues](https://github.com/cogpy/rwkv-raven-cog/issues)**
4. **Create a new issue** with detailed error information
5. **Join the community** discussions for help and tips

Congratulations! You're now ready to explore the fascinating intersection of neural language models and cognitive architectures with RWKV-Raven-Cog. 🎉