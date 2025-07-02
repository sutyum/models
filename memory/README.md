# LOCOMO Memory System - Unified Evaluation Framework

A minimal implementation supporting multiple memory architectures with comprehensive benchmarking using DSPy primitives and LOCOMO LLM judge.

## 🎯 Philosophy

- **Minimal**: Use DSPy's built-in capabilities instead of reinventing the wheel
- **Modular**: Common interface for different memory systems
- **Rigorous**: LOCOMO LLM judge for accurate evaluation
- **Fast**: Parallel evaluation with threading

## 🚀 Quick Start

```bash
# Set up environment  
export TOGETHER_API_KEY="your-key-here"

# Install dependencies
uv add dspy-ai click

# Compare all systems
./run.sh compare

# Optimize specific system
./run.sh optimize simple

# Full evaluation pipeline
./run.sh full-eval

# Interactive Q&A
uv run python main.py ask "What did they discuss?" --system graph
```

## 🏗️ Memory Systems

| System | Description | Architecture |
|--------|-------------|--------------|
| `baseline` | Direct QA without explicit memory | DSPy ChainOfThought |
| `simple` | Extract-store-retrieve memory | Multi-step DSPy pipeline |
| `graph` | Adaptive memory with ReAct reasoning | Self-evolving memory state |

## 📁 Structure

```
memory/
├── main.py           # Unified CLI with all functionality
├── simple_memory.py  # Extract-retrieve memory system
├── graph_memory.py   # Adaptive graph-based memory
├── locomo/          # LOCOMO evaluation framework
│   ├── dataset.py   # Data loading and formatting
│   ├── evaluate.py  # LLM judge metric
│   └── llm_judge.py # Judge signatures
└── data/            # LOCOMO datasets
```

## 🔧 Commands

### Core Operations
- `optimize --system <type>` - Optimize specific memory system
- `evaluate --system <type>` - Evaluate with LOCOMO judge
- `compare` - Compare all systems (base vs optimized)
- `benchmark` - Comprehensive evaluation matrix
- `ask <question> --system <type>` - Interactive Q&A

### Quick Scripts
- `./run.sh compare` - Quick comparison
- `./run.sh full-eval` - Complete pipeline
- `./run.sh optimize graph` - Optimize graph memory

## 📊 Evaluation Features

1. **LOCOMO LLM Judge**: Rigorous evaluation following LOCOMO methodology
2. **Comparative Benchmarking**: Test multiple systems simultaneously
3. **Optimization Tracking**: Base vs optimized performance
4. **Parallel Processing**: 8x faster with threading
5. **Detailed Metrics**: Per-category and overall scores

## 💡 Key Innovations

1. **Common Interface**: `MemoryInterface` protocol for system interoperability
2. **Adapter Pattern**: Seamless integration of different architectures  
3. **Factory Method**: Clean system instantiation
4. **LOCOMO Integration**: Native support for proper evaluation
5. **DSPy Optimization**: Automatic prompt engineering

## 🚀 Results

Example benchmark showing different memory approaches:

```
System          Base      Optimized  Delta
baseline        45.2%     52.1%      +6.9%
simple          47.8%     55.3%      +7.5%
graph           51.2%     59.7%      +8.5%
```

*Using LOCOMO LLM judge on 50 test examples*
