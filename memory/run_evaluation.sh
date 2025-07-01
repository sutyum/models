#!/bin/bash
set -e

echo "LOCOMO Benchmark Evaluation with Optimized Prompt"
echo "================================================="

# Check if optimized prompt file exists
if [ ! -f "optimized_memory_qa.json" ]; then
    echo "Error: optimized_memory_qa.json not found!"
    echo "Please ensure the optimized prompt file is in the current directory."
    exit 1
fi

# Check if test data exists
if [ ! -f "data/locomo_test.json" ]; then
    echo "Error: data/locomo_test.json not found!"
    echo "Please ensure the LOCOMO test dataset is available."
    exit 1
fi

# Set up environment variables if needed
export OPENAI_API_KEY=${OPENAI_API_KEY:-""}

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY not set. Please set it for OpenAI models."
    echo "export OPENAI_API_KEY='your-api-key'"
fi

# Create results directory
mkdir -p results

# Get timestamp for results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="results/locomo_evaluation_${TIMESTAMP}.json"

echo "Starting evaluation..."
echo "- Optimized prompt: optimized_memory_qa.json"
echo "- Test data: data/locomo_test.json"
echo "- Results will be saved to: $RESULTS_FILE"
echo ""

# Run evaluation with different models if available
MODELS=("gpt-3.5-turbo" "gpt-4" "gpt-4-turbo")
SELECTED_MODEL="gpt-3.5-turbo"

# Check if user wants to specify a model
if [ $# -gt 0 ]; then
    SELECTED_MODEL=$1
    echo "Using specified model: $SELECTED_MODEL"
else
    echo "Using default model: $SELECTED_MODEL"
    echo "(You can specify a different model as an argument: ./run_evaluation.sh gpt-4)"
fi

echo ""

# Run the evaluation
python evaluate_optimized.py \
    --optimized-prompt optimized_memory_qa.json \
    --test-data data/locomo_test.json \
    --output "$RESULTS_FILE" \
    --model "$SELECTED_MODEL"

# Check if evaluation was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "Evaluation completed successfully!"
    echo "Results saved to: $RESULTS_FILE"
    
    # Display quick summary if jq is available
    if command -v jq &> /dev/null; then
        echo ""
        echo "Quick Summary:"
        echo "=============="
        jq -r '
        "Overall Score: " + (.overall_llm_judge_score | tostring) + 
        "\nTotal Examples: " + (.total_examples | tostring) + 
        "\nEvaluation Time: " + (.evaluation_time_seconds | tostring) + "s"
        ' "$RESULTS_FILE"
        
        echo ""
        echo "Category Breakdown:"
        jq -r '
        range(1; 6) as $i | 
        "Category \($i): " + 
        (if .["category_\($i)_count"] > 0 then 
            (.["category_\($i)_score"] | tostring) + " (" + (.["category_\($i)_count"] | tostring) + " examples)"
        else 
            "No examples"
        end)
        ' "$RESULTS_FILE"
    fi
else
    echo "Evaluation failed. Check the output above for errors."
    exit 1
fi
