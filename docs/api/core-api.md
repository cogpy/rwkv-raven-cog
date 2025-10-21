# Core API Reference

This document provides comprehensive API documentation for the core classes and functions in RWKV-Raven-Cog, including detailed parameter descriptions, return values, and usage examples.

## 📚 Table of Contents

- [OpenCogTransformer](#opencogtransformer) - Main transformation controller
- [OpenCogConfig](#opencogconfig) - Configuration management
- [CognitiveLayer](#cognitivelayer) - Cognitive module representation
- [AtomspaceNode](#atomspacenode) - Knowledge representation unit
- [Utility Functions](#utility-functions) - Helper functions and utilities

## 🎛️ OpenCogTransformer

The main controller class that orchestrates the transformation of RWKV models into OpenCog-compatible cognitive architectures.

### Class Definition

```python
class OpenCogTransformer:
    """
    Main transformer class for RWKV to OpenCog conversion
    
    This class handles the complete transformation pipeline from loading RWKV models
    to generating OpenCog-compatible cognitive architectures with symbolic reasoning
    capabilities.
    
    Attributes:
        config (OpenCogConfig): Transformation configuration parameters
        cognitive_layers (Dict[str, CognitiveLayer]): Generated cognitive modules
        atomspace (List[AtomspaceNode]): Symbolic knowledge representation
        attention_map (Dict[str, float]): Attention allocation network
    """
```

### Constructor

```python
def __init__(self, config: OpenCogConfig):
    """
    Initialize the OpenCog transformer
    
    Args:
        config (OpenCogConfig): Configuration object containing transformation parameters
                              including atomspace size, attention thresholds, and 
                              cognitive module specifications
    
    Raises:
        ValueError: If configuration parameters are invalid
        TypeError: If config is not an OpenCogConfig instance
    
    Example:
        >>> config = OpenCogConfig(atomspace_size=15000, attention_threshold=0.6)
        >>> transformer = OpenCogTransformer(config)
    """
```

### Core Methods

#### load_rwkv_model

```python
def load_rwkv_model(self, model_path: str) -> Dict[str, Any]:
    """
    Load RWKV model from file path with comprehensive error handling
    
    Supports PyTorch .pth files and provides fallback to mock model structure
    for testing and demonstration purposes when actual models are unavailable.
    
    Args:
        model_path (str): Path to RWKV model file (supports .pth format)
    
    Returns:
        Dict[str, Any]: Model state dictionary containing:
            - 'emb': Embedding layer weights and configuration
            - 'blocks': Transformer block parameters (attention + FFN)  
            - 'ln_out': Final layer normalization parameters
            - 'head': Output projection layer parameters
    
    Raises:
        FileNotFoundError: If model file doesn't exist and mock fallback fails
        RuntimeError: If model loading encounters unexpected errors
        
    Example:
        >>> model_dict = transformer.load_rwkv_model('model.pth')
        >>> print(f"Loaded model with {len(model_dict)} components")
    """
```

#### transform_layers

```python
def transform_layers(self, model_state_dict: Dict[str, Any]) -> Dict[str, CognitiveLayer]:
    """
    Transform RWKV layers into cognitive module representations
    
    Maps neural network components to cognitive functions:
    - Embedding → Perception (input processing)
    - Attention → Attention (focus management) 
    - Feed-forward → Reasoning (inference processing)
    - Output → Language (communication generation)
    
    Args:
        model_state_dict (Dict[str, Any]): RWKV model parameters from load_rwkv_model
    
    Returns:
        Dict[str, CognitiveLayer]: Cognitive layers mapped by name, each containing:
            - layer_name: Unique identifier for the cognitive layer
            - module_type: Cognitive function category (perception/attention/reasoning/language)
            - input_size: Input dimension size
            - output_size: Output dimension size  
            - pattern_templates: Symbolic reasoning patterns for this layer
            - atomspace_nodes: Associated knowledge representation nodes
    
    Example:
        >>> cognitive_layers = transformer.transform_layers(model_dict)
        >>> for name, layer in cognitive_layers.items():
        ...     print(f"{name}: {layer.module_type} ({layer.input_size}→{layer.output_size})")
    """
```

#### create_attention_network

```python
def create_attention_network(self, layers: Dict[str, CognitiveLayer]) -> Dict[str, float]:
    """
    Create dynamic attention allocation network for cognitive resource management
    
    Calculates attention values based on:
    - Module type importance (perception=0.8, attention=0.9, reasoning=0.7, language=0.8)
    - Layer complexity and connectivity
    - Processing requirements and resource constraints
    
    Args:
        layers (Dict[str, CognitiveLayer]): Cognitive layers from transform_layers
    
    Returns:
        Dict[str, float]: Attention allocation map with values in [0.1, 1.0]:
            - Keys: Layer names matching cognitive_layers keys
            - Values: Attention weights indicating processing priority
            - Higher values indicate greater resource allocation priority
    
    Example:
        >>> attention_map = transformer.create_attention_network(cognitive_layers)
        >>> sorted_attention = sorted(attention_map.items(), key=lambda x: x[1], reverse=True)
        >>> print(f"Highest attention: {sorted_attention[0]}")
    """
```

#### generate_atomspace_representation

```python
def generate_atomspace_representation(self, layers: Dict[str, CognitiveLayer]) -> List[AtomspaceNode]:
    """
    Generate comprehensive Atomspace knowledge representation
    
    Creates symbolic nodes and relationships representing:
    - Conceptual layer functions and module types
    - Hierarchical cognitive architecture structure  
    - Pattern matching templates for symbolic reasoning
    - Attention and inference relationship links
    
    Args:
        layers (Dict[str, CognitiveLayer]): Cognitive layers with pattern templates
    
    Returns:
        List[AtomspaceNode]: Complete symbolic knowledge representation containing:
            - ConceptNode: Basic cognitive concepts and layer representations
            - SchemaNode: Pattern matching templates and reasoning procedures  
            - Connection objects: Relationships between concepts and schemas
            - Truth values: Confidence and strength assessments
    
    Example:
        >>> atomspace_nodes = transformer.generate_atomspace_representation(layers)
        >>> concept_nodes = [n for n in atomspace_nodes if n.atom_type == 'ConceptNode']
        >>> print(f"Generated {len(concept_nodes)} concept nodes")
    """
```

#### transform_model

```python
def transform_model(self, model_path: str, output_dir: str) -> Dict[str, Any]:
    """
    Complete transformation pipeline from RWKV model to OpenCog configuration
    
    Orchestrates the full transformation process:
    1. Load and validate RWKV model
    2. Analyze and transform neural layers  
    3. Generate cognitive module representations
    4. Create attention allocation network
    5. Build symbolic knowledge representation
    6. Export complete OpenCog configuration
    
    Args:
        model_path (str): Path to RWKV model file (.pth format)
        output_dir (str): Directory for output files (created if doesn't exist)
    
    Returns:
        Dict[str, Any]: Complete transformation results containing:
            - 'cognitive_layers': Generated cognitive module representations
            - 'attention_map': Resource allocation network
            - 'atomspace_nodes': Symbolic knowledge representation  
            - 'config_path': Path to generated configuration file
            - 'metadata': Transformation process information
    
    Raises:
        IOError: If unable to create output directory or write files
        RuntimeError: If transformation pipeline encounters fatal errors
        
    Example:
        >>> result = transformer.transform_model('model.pth', 'output/')  
        >>> print(f"Generated {len(result['cognitive_layers'])} cognitive layers")
        >>> print(f"Configuration saved to: {result['config_path']}")
    """
```

## ⚙️ OpenCogConfig

Configuration management class for transformation parameters and cognitive architecture settings.

### Class Definition

```python
@dataclass
class OpenCogConfig:
    """
    Configuration for OpenCog transformation with validation and defaults
    
    Manages all parameters controlling the RWKV to OpenCog transformation process,
    including knowledge representation size, attention mechanisms, reasoning depth,
    and cognitive module specifications.
    """
```

### Configuration Parameters

```python
# Core Configuration
atomspace_size: int = 10000
    """Maximum number of atoms in the knowledge base
    
    Controls the size of the symbolic knowledge representation.
    Larger values allow more detailed representations but require more memory.
    
    Range: 1000-100000 (recommended: 10000-50000)
    """

attention_threshold: float = 0.5  
    """Minimum attention value for active cognitive processes
    
    Filters low-importance cognitive processes to focus computational resources.
    Higher values create more selective attention but may miss subtle patterns.
    
    Range: 0.1-0.9 (recommended: 0.3-0.7)
    """

pattern_match_depth: int = 3
    """Maximum depth for pattern matching and symbolic reasoning
    
    Controls how many levels deep the pattern matching engine searches.
    Deeper searches find more complex patterns but take longer to compute.
    
    Range: 1-10 (recommended: 3-6)
    """

cognitive_modules: List[str] = None
    """List of cognitive module types to include in transformation
    
    Specifies which cognitive functions to model. Available modules:
    - 'perception': Input processing and feature extraction
    - 'attention': Focus management and resource allocation
    - 'reasoning': Logical inference and problem solving  
    - 'language': Communication and expression generation
    - 'memory': Knowledge storage and retrieval
    - 'emotion': Affective processing and motivation
    - 'action': Motor control and behavioral execution
    - 'learning': Adaptation and skill acquisition
    
    Default: ['perception', 'attention', 'reasoning', 'language', 'memory']
    """

enable_symbolic_integration: bool = True
    """Enable symbolic reasoning integration with neural processing
    
    When True, creates bidirectional mappings between neural activations
    and symbolic representations for hybrid reasoning capabilities.
    """

memory_optimization: bool = True  
    """Enable memory-efficient processing optimizations
    
    Applies various optimizations to reduce memory usage during transformation:
    - Lazy loading of model components
    - Streaming processing for large models
    - Garbage collection of intermediate representations
    """
```

### Methods

```python
def validate(self) -> bool:
    """
    Validate configuration parameters for consistency and feasibility
    
    Checks parameter ranges, dependencies, and resource requirements.
    
    Returns:
        bool: True if configuration is valid, False otherwise
    
    Raises:
        ValueError: If parameters are outside valid ranges or incompatible
    """

def get_memory_estimate(self) -> int:
    """
    Estimate memory requirements for this configuration
    
    Returns:
        int: Estimated memory usage in bytes
    """

def save(self, filepath: str):
    """Save configuration to JSON file"""
    
def load(self, filepath: str) -> 'OpenCogConfig':
    """Load configuration from JSON file"""
```

### Usage Examples

```python
# Basic configuration
config = OpenCogConfig()

# High-performance configuration  
config = OpenCogConfig(
    atomspace_size=50000,
    attention_threshold=0.7,
    pattern_match_depth=5,
    memory_optimization=True
)

# Research configuration with all modules
config = OpenCogConfig(
    atomspace_size=25000,
    cognitive_modules=[
        'perception', 'attention', 'reasoning', 'language', 
        'memory', 'emotion', 'action', 'learning'
    ],
    enable_symbolic_integration=True,
    pattern_match_depth=4
)
```

## 🧠 CognitiveLayer

Represents individual cognitive modules in the transformed architecture.

### Class Definition

```python
class CognitiveLayer:
    """
    Represents a cognitive layer in the OpenCog transformation
    
    Encapsulates the functional and structural properties of cognitive modules,
    including their neural substrate, symbolic representations, processing
    characteristics, and integration with other cognitive components.
    """
```

### Constructor and Attributes

```python
def __init__(self, layer_name: str, input_size: int, output_size: int, 
             module_type: str = 'general'):
    """
    Initialize cognitive layer with structural and functional properties
    
    Args:
        layer_name (str): Unique identifier for this cognitive layer
        input_size (int): Input dimension size (number of input features)
        output_size (int): Output dimension size (number of output features)  
        module_type (str): Cognitive function type from available modules
    """

# Core attributes
layer_name: str              # Unique layer identifier
input_size: int              # Input feature dimensions  
output_size: int             # Output feature dimensions
module_type: str             # Cognitive function category
atomspace_nodes: List[AtomspaceNode]  # Symbolic representations
pattern_templates: List[str]          # Reasoning pattern templates
attention_value: float                # Current attention allocation
activation_level: float               # Current processing activity
```

### Core Methods

```python
def add_pattern_template(self, template: str):
    """
    Add symbolic reasoning pattern template to this layer
    
    Templates define symbolic structures for pattern matching and inference.
    
    Args:
        template (str): Pattern template in Scheme-like syntax
    
    Example:
        >>> layer.add_pattern_template(
        ...     "(EvaluationLink (PredicateNode 'processes') $input)"
        ... )
    """

def create_atomspace_representation(self, tensor_data: torch.Tensor) -> List[AtomspaceNode]:
    """
    Convert tensor data to symbolic Atomspace node representation
    
    Creates symbolic nodes representing the neural activation patterns
    in this layer, enabling symbolic reasoning over neural representations.
    
    Args:
        tensor_data (torch.Tensor): Neural activation data from this layer
    
    Returns:
        List[AtomspaceNode]: Symbolic nodes representing tensor information
    """

def get_activation_level(self) -> float:
    """Get current activation level of this cognitive layer"""

def update_attention_value(self, new_attention: float):
    """Update attention allocation for this layer"""

def get_processing_statistics(self) -> Dict[str, Any]:
    """Get detailed processing statistics for this layer"""
```

## ⚛️ AtomspaceNode

Basic unit of symbolic knowledge representation in the OpenCog Atomspace.

### Class Definition

```python
class AtomspaceNode:
    """
    Represents a node in OpenCog's Atomspace knowledge representation
    
    Encodes symbolic concepts, relationships, and reasoning patterns with
    associated truth values and attention mechanisms for dynamic knowledge
    processing and symbolic-subsymbolic integration.
    """
```

### Constructor and Attributes

```python  
def __init__(self, atom_type: str, name: str, truth_value: float = 1.0):
    """
    Initialize Atomspace node with symbolic properties
    
    Args:
        atom_type (str): Type of atom ('ConceptNode', 'PredicateNode', 'SchemaNode', etc.)
        name (str): Unique name identifier for this atom
        truth_value (float): Initial truth strength in [0.0, 1.0]
    """

# Core attributes
atom_type: str               # Node type (ConceptNode, PredicateNode, etc.)
name: str                    # Unique identifier string
truth_value: float           # Truth strength [0.0, 1.0] 
confidence: float            # Truth confidence [0.0, 1.0]
attention_value: float       # Current attention allocation
connections: List[Dict]      # Relationships to other nodes
creation_time: datetime      # Timestamp of node creation
last_accessed: datetime      # Last access timestamp
access_count: int            # Number of times accessed
```

### Core Methods

```python
def add_connection(self, target_node: 'AtomspaceNode', link_type: str = 'SimilarityLink'):
    """
    Add relationship connection to another Atomspace node
    
    Creates directed or undirected links between concepts for relationship
    representation and graph-based reasoning.
    
    Args:
        target_node (AtomspaceNode): Target node for the relationship
        link_type (str): Type of relationship link
    
    Available link types:
        - 'SimilarityLink': Semantic similarity relationship
        - 'InheritanceLink': Is-a hierarchical relationship  
        - 'MemberLink': Set membership relationship
        - 'EvaluationLink': Predicate evaluation relationship
        - 'AttentionLink': Attention allocation relationship
    """

def update_truth_value(self, new_strength: float, new_confidence: float = None):
    """Update truth value with evidence integration"""

def calculate_similarity(self, other_node: 'AtomspaceNode') -> float:
    """Calculate semantic similarity with another node"""

def get_connected_nodes(self, link_type: str = None) -> List['AtomspaceNode']:
    """Get all nodes connected via specified link type"""

def to_scheme_representation(self) -> str:
    """Convert node to Scheme-format representation for OpenCog"""
```

## 🔧 Utility Functions

### Model Analysis Functions

```python
def analyze_model_complexity(model_dict: Dict[str, Any]) -> Dict[str, int]:
    """
    Analyze RWKV model complexity and structure
    
    Args:
        model_dict: RWKV model state dictionary
    
    Returns:
        Dict containing:
        - 'total_parameters': Total parameter count
        - 'layer_count': Number of transformer layers
        - 'embedding_size': Embedding dimension  
        - 'vocabulary_size': Token vocabulary size
        - 'attention_heads': Number of attention mechanisms
    """

def estimate_transformation_time(model_dict: Dict[str, Any], 
                               config: OpenCogConfig) -> float:
    """Estimate transformation processing time in seconds"""

def validate_model_compatibility(model_dict: Dict[str, Any]) -> bool:
    """Check if RWKV model is compatible with transformation"""
```

### Configuration Utilities

```python
def create_default_config(model_size: str = "3B") -> OpenCogConfig:
    """
    Create optimized configuration for specific model sizes
    
    Args:
        model_size: Model size specification ("1B5", "3B", "7B", "14B")
    
    Returns:
        OpenCogConfig: Optimized configuration for the model size
    """

def merge_configs(base_config: OpenCogConfig, 
                 override_config: OpenCogConfig) -> OpenCogConfig:
    """Merge two configurations with override precedence"""

def validate_config_compatibility(config: OpenCogConfig, 
                                system_resources: Dict) -> List[str]:
    """Validate configuration against system resource constraints"""
```

### Export and Serialization

```python
def export_to_opencog_format(atomspace_nodes: List[AtomspaceNode], 
                           output_path: str):
    """Export Atomspace nodes to OpenCog-compatible format"""

def generate_visualization_data(cognitive_layers: Dict[str, CognitiveLayer],
                              attention_map: Dict[str, float]) -> Dict:
    """Generate data for visualization tools and dashboards"""

def create_integration_bridge(transformation_result: Dict[str, Any]) -> Dict:
    """Create integration bridge for connecting to OpenCog systems"""
```

### Error Handling and Logging

```python
class TransformationError(Exception):
    """Custom exception for transformation-related errors"""

class ConfigurationError(Exception):  
    """Custom exception for configuration-related errors"""

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Configure logging for transformation operations"""

def get_system_diagnostics() -> Dict[str, Any]:
    """Get comprehensive system diagnostic information"""
```

This comprehensive API reference provides detailed documentation for all core components, enabling effective use and extension of the RWKV-Raven-Cog system.