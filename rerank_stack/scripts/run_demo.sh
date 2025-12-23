#!/bin/bash
# Script to run the rerank demo application

# Navigate to the rerank_stack root directory
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR/.." || exit 1

# Activate the virtual environment
# source venv/bin/activate # No longer strictly needed for python call

# Set PYTHONPATH to include the src directory
export PYTHONPATH=src

# Check if an embedding model is provided, default to nomic-embed-text
EMBEDDING_MODEL=${1:-"nomic-embed-text"}

echo "--- Building Embedding Index ---"
./venv/bin/python -m rerank_demo.build_index --embedding_model $EMBEDDING_MODEL

echo ""
echo "--- Running Demo Search and Rerank ---"
# Example query
./venv/bin/python -m rerank_demo.search "shower repair"

echo ""
echo "--- Demo Complete ---"
echo "You can modify the search.py or build_index.py for different queries or models."
