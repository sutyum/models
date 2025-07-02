#!/bin/bash
# LOCOMO Memory System Evaluation Runner

set -e

# Check environment
if [ -z "$TOGETHER_API_KEY" ]; then
    echo "Error: Set TOGETHER_API_KEY environment variable"
    exit 1
fi

# Default command and system
CMD=${1:-compare}
SYSTEM=${2:-baseline}

case $CMD in
    optimize)
        echo "Optimizing ${SYSTEM} memory system..."
        uv run python main.py optimize --system "$SYSTEM" --limit 20
        ;;
    evaluate)
        echo "Evaluating ${SYSTEM} memory system..."
        uv run python main.py evaluate --system "$SYSTEM" --limit 10
        ;;
    compare)
        echo "Comparing all memory systems..."
        uv run python main.py compare --limit 10
        ;;
    benchmark)
        echo "Running comprehensive benchmark..."
        uv run python main.py benchmark --limit 20
        ;;
    ask)
        echo "Interactive Q&A mode with ${SYSTEM}..."
        uv run python main.py ask "What did they discuss?" --system "$SYSTEM"
        ;;
    full-eval)
        echo "Full evaluation pipeline..."
        for sys in baseline simple graph; do
            echo "Optimizing $sys..."
            uv run python main.py optimize --system "$sys" --limit 50
        done
        echo "Running comparison..."
        uv run python main.py compare --limit 50
        ;;
    *)
        echo "Usage: $0 [optimize|evaluate|compare|benchmark|ask|full-eval] [system]"
        echo ""
        echo "Commands:"
        echo "  optimize    - Optimize a specific memory system"
        echo "  evaluate    - Evaluate a specific memory system"
        echo "  compare     - Compare all systems (base vs optimized)"
        echo "  benchmark   - Comprehensive benchmark"
        echo "  ask         - Interactive Q&A"
        echo "  full-eval   - Complete evaluation pipeline"
        echo ""
        echo "Systems: baseline, simple, graph"
        exit 1
        ;;
esac