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

# Compare all systems (quick)
./run.sh compare

# Optimize specific system
./run.sh optimize simple

# Full evaluation pipeline
./run.sh full-eval

# Direct usage examples
uv run python main.py compare --limit 5
uv run python main.py optimize --system graph --limit 20
uv run python main.py evaluate --system simple --limit 10
uv run python main.py ask "What did they discuss?" --system baseline
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
```bash
# Optimize with fast SIMBA (parallel)
uv run python main.py optimize --system graph --method simba --threads 8 --limit 50

# Optimize with other methods
uv run python main.py optimize --system simple --method bootstrap --limit 30
uv run python main.py optimize --system baseline --method mipro --limit 20

# Evaluate with LOCOMO judge
uv run python main.py evaluate --system simple --limit 20

# Compare all systems (base vs optimized)
uv run python main.py compare --limit 10

# Comprehensive evaluation matrix
uv run python main.py benchmark --limit 20

# Interactive Q&A
uv run python main.py ask "What was discussed?" --system baseline
```

### Quick Scripts
```bash
# Quick comparison (10 examples)
./run.sh compare

# Complete evaluation pipeline
./run.sh full-eval

# Optimize specific system
./run.sh optimize graph

# Interactive Q&A mode
./run.sh ask
```

## 📊 Evaluation Features

1. **LOCOMO LLM Judge**: Rigorous evaluation following LOCOMO methodology
2. **Comparative Benchmarking**: Test multiple systems simultaneously
3. **Optimization Tracking**: Base vs optimized performance
4. **Parallel Processing**: 8x faster with threading
5. **Detailed Metrics**: Per-category and overall scores

## ⚡ Performance Optimization

### Fast Optimization with SIMBA
SIMBA is the fastest optimization method with built-in parallelization:

```bash
# Fast parallel optimization (8 threads)
uv run python main.py optimize --method simba --threads 8 --limit 50

# Even faster with fewer steps
uv run python main.py optimize --method simba --threads 16 --limit 30
```

### Optimization Method Comparison
| Method | Speed | Quality | Best For |
|--------|-------|---------|----------|
| `simba` | ⚡⚡⚡ Fast | High | Production use |
| `bootstrap` | ⚡⚡ Medium | Medium | Quick prototyping |
| `mipro` | ⚡ Slow | Highest | Research/best results |

### Threading Guidelines
- **Optimization**: Use 8-16 threads for SIMBA
- **Evaluation**: Use 8 threads (default)
- **Memory**: Graph memory is slowest, baseline fastest

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

## 🔧 Usage Notes

### Shell Script Usage
```bash
# Correct - run shell script directly
./run.sh compare

# Incorrect - don't use uv run with shell scripts
# uv run ./run.sh compare  # This will fail
```

### Python Commands
```bash
# Always use uv run for Python commands
uv run python main.py compare --limit 5
uv run python main.py optimize --system simple
```

### Troubleshooting
1. **Permission Error**: `chmod +x run.sh`
2. **Missing Dependencies**: `uv add dspy-ai click`
3. **API Key**: `export TOGETHER_API_KEY="your-key"`
4. **Data Files**: Ensure `data/locomo_*.json` files exist

### Environment Setup
```bash
# One-time setup
export TOGETHER_API_KEY="your-together-ai-key"
uv add dspy-ai click

# Verify installation
uv run python main.py --help
```
