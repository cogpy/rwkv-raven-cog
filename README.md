# RWKV-Raven-Cog: OpenCog Transformation for RWKV Models

This repository implements OpenCog cognitive architecture transformations for RWKV-4-Raven language models. The transformation restructures RWKV model layers and tensors to integrate with OpenCog's symbolic-subsymbolic AI framework.

## Overview

RWKV (Receptance Weighted Key Value) is a novel architecture that combines the benefits of RNNs and Transformers. This project transforms RWKV models to work within OpenCog's cognitive architecture, enabling:

- **Symbolic Integration**: RWKV layers mapped to OpenCog Atomspace representations
- **Cognitive Modules**: Layer restructuring for perception, attention, reasoning, and language modules
- **Pattern Matching**: Integration with OpenCog's pattern matching capabilities
- **Attention Allocation**: Dynamic attention management across cognitive processes

## Features

- 🧠 **Cognitive Layer Transformation**: Convert RWKV blocks to specialized cognitive modules
- 🔗 **Atomspace Integration**: Map model tensors to OpenCog's knowledge representation
- 🎯 **Attention Networks**: Create attention allocation maps for cognitive resource management
- 📊 **Pattern Templates**: Generate pattern matching templates for symbolic reasoning
- ⚡ **Memory Optimization**: Efficient handling of large model transformations

## Installation

### Prerequisites

Make sure you have the following installed:
- Python 3.8+
- Git LFS: https://git-lfs.com
- HuggingFace CLI: `pip install -U "huggingface_hub[cli]"`

### Quick Setup

```bash
# Clone this repository
git clone https://github.com/cogpy/rwkv-raven-cog.git
cd rwkv-raven-cog

# Run the setup script
chmod +x setup_environment.sh
./setup_environment.sh

# Or install manually:
pip install -r requirements.txt
git lfs install
```

### Install as Package

```bash
pip install -e .
```

## Usage

### Command Line Interface

```bash
# Run the transformation script directly
python opencog_transform.py

# Or if installed as package
rwkv-opencog-transform
```

### Python API

```python
from opencog_transform import OpenCogTransformer, OpenCogConfig

# Configure transformation
config = OpenCogConfig(
    atomspace_size=15000,
    attention_threshold=0.6,
    pattern_match_depth=4,
    enable_symbolic_integration=True
)

# Initialize transformer
transformer = OpenCogTransformer(config)

# Transform a model
result = transformer.transform_model(
    model_path='rwkv-4-raven/RWKV-4-Raven-3B-v12-Eng98%25-Other2%25-20230520-ctx4096.pth',
    output_dir='opencog_transformed/3B_model'
)
```

## RWKV Model Acquisition

The transformation works with RWKV-4-Raven models from HuggingFace. To download models:

### Method 1: Git Clone (with LFS skip for efficiency)
```bash
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/BlinkDL/rwkv-4-raven
```

### Method 2: HuggingFace CLI
```bash
hf download BlinkDL/rwkv-4-raven
```

## Architecture

### 📚 Comprehensive Technical Documentation

For detailed technical architecture documentation with extensive Mermaid diagrams, see:

- **[📖 Documentation Hub](docs/README.md)** - Complete documentation navigation
- **[🏗️ System Overview](docs/architecture/system-overview.md)** - High-level architecture with component diagrams
- **[🔄 Transformation Pipeline](docs/architecture/transformation-pipeline.md)** - Detailed 7-stage transformation process
- **[🧠 Cognitive Layers](docs/architecture/cognitive-layers.md)** - Cognitive module architecture and interactions
- **[⚡ Attention Network](docs/architecture/attention-network.md)** - Dynamic resource allocation system
- **[⚛️ Atomspace Integration](docs/architecture/atomspace-integration.md)** - Symbolic knowledge representation
- **[📊 System Flow Diagrams](docs/diagrams/system-flow.md)** - Visual data and control flow
- **[🔧 Core API Reference](docs/api/core-api.md)** - Complete API documentation
- **[🚀 Quick Start Guide](docs/guides/quick-start.md)** - Get up and running in minutes

### OpenCog Integration

The transformation maps RWKV components to OpenCog cognitive modules:

- **Embedding Layer** → **Perception Module**: Word/token perception and encoding
- **Attention Blocks** → **Attention Module**: Cognitive attention allocation 
- **Feed-Forward Networks** → **Reasoning Module**: Symbolic-subsymbolic reasoning
- **Output Layer** → **Language Module**: Token generation and language production

### Cognitive Layer Structure

Each cognitive layer includes:
- **Atomspace Nodes**: Conceptual representations of layer functions
- **Pattern Templates**: Symbolic patterns for reasoning and inference
- **Attention Values**: Dynamic attention allocation weights
- **Connection Maps**: Links between cognitive components

### Transformation Pipeline

1. **Model Loading**: Load RWKV model weights and configuration
2. **Layer Analysis**: Analyze RWKV layer structure and dimensions
3. **Cognitive Mapping**: Map layers to cognitive module types
4. **Atomspace Creation**: Generate Atomspace node representations
5. **Pattern Generation**: Create symbolic pattern matching templates
6. **Attention Network**: Build attention allocation network
7. **Configuration Export**: Save transformation configuration

## Output Structure

The transformation generates:

```
opencog_transformed/
├── {model_name}/
│   ├── opencog_config.json      # Transformation configuration
│   ├── cognitive_layers.json    # Layer mapping details
│   ├── atomspace_nodes.json     # Atomspace representations
│   └── attention_network.json   # Attention allocation map
```

## Configuration

### OpenCogConfig Parameters

- `atomspace_size`: Maximum number of atoms in the knowledge base
- `attention_threshold`: Minimum attention value for active nodes  
- `pattern_match_depth`: Maximum depth for pattern matching
- `cognitive_modules`: List of cognitive module types to create
- `enable_symbolic_integration`: Enable symbolic reasoning integration
- `memory_optimization`: Enable memory-efficient processing

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests (when implemented)
pytest tests/
```

### Code Style

```bash
# Format code
black opencog_transform.py

# Lint code  
flake8 opencog_transform.py
```

## Supported Models

Currently supports RWKV-4-Raven models:
- RWKV-4-Raven-1B5 (1.5B parameters)
- RWKV-4-Raven-3B (3B parameters)  
- RWKV-4-Raven-7B (7B parameters)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [RWKV Project](https://github.com/BlinkDL/RWKV-LM) for the innovative architecture
- [OpenCog Foundation](https://opencog.org/) for the cognitive architecture framework
- [HuggingFace](https://huggingface.co/) for model hosting and tools

## Citation

If you use this work in your research, please cite:

```bibtex
@software{rwkv_raven_cog,
  title={RWKV-Raven-Cog: OpenCog Transformation for RWKV Models},
  author={OpenCog Community},
  year={2024},
  url={https://github.com/cogpy/rwkv-raven-cog}
}
```