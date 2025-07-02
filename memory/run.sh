#!/bin/bash
# LOCOMO Memory System Evaluation Runner

set -e

# Check environment
if [ -z "$TOGETHER_API_KEY" ] && [ -z "$GROQ_API_KEY" ]; then
    echo "Error: Set TOGETHER_API_KEY or GROQ_API_KEY environment variable"
    exit 1
fi

# Default command, system, and model
CMD=${1:-compare}
SYSTEM=${2:-baseline}
MODEL=${3:-gemini_flash}

case $CMD in
    optimize)
        echo "Optimizing ${SYSTEM} memory system with ${MODEL}..."
        uv run python main.py optimize --system "$SYSTEM" --model "$MODEL" --method simba --threads 8 --limit 20
        ;;
    evaluate)
        echo "Evaluating ${SYSTEM} memory system with ${MODEL}..."
        uv run python main.py evaluate --system "$SYSTEM" --model "$MODEL" --limit 10
        ;;
    compare)
        echo "Comparing all memory systems with ${MODEL}..."
        uv run python main.py compare --model "$MODEL" --limit 10
        ;;
    benchmark)
        echo "Running comprehensive benchmark with ${MODEL}..."
        uv run python main.py benchmark --model "$MODEL" --limit 20
        ;;
    ask)
        echo "Interactive Q&A mode with ${SYSTEM} using ${MODEL}..."
        uv run python main.py ask "What did they discuss?" --system "$SYSTEM" --model "$MODEL"
        ;;
    full-eval)
        echo "Full evaluation pipeline with ${MODEL}..."
        for sys in baseline simple graph; do
            echo "Optimizing $sys..."
            uv run python main.py optimize --system "$sys" --model "$MODEL" --method simba --threads 8 --limit 50
        done
        echo "Running comparison..."
        uv run python main.py compare --model "$MODEL" --limit 50
        ;;
    *)
        echo "Usage: $0 [optimize|evaluate|compare|benchmark|ask|full-eval] [system] [model]"
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
        echo "Models: qwen, deepseek, llama, mixtral, gemma"
        echo ""
        echo "Examples:"
        echo "  $0 compare baseline llama    # Compare with Llama on Groq"
        echo "  $0 optimize graph mixtral    # Optimize graph memory with Mixtral"
        exit 1
        ;;
esac
