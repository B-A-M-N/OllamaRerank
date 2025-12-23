#!/bin/bash
# Script to run the rerank API service

# Navigate to the rerank_stack root directory
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR/.." || exit 1

# Activate the virtual environment
# source venv/bin/activate # No longer strictly needed for uvicorn call

# Set PYTHONPATH to include the src directory
export PYTHONPATH=src

# Define absolute paths for log files
LOG_DIR="/home/bamn/OllamaRerank/.gemini/tmp"
SERVER_OUT_LOG="$LOG_DIR/server_out.log"
SERVER_ERR_LOG="$LOG_DIR/server_err.log"

# Create the log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Run the FastAPI application using explicit path
./venv/bin/uvicorn rerank_service.api:app --port 8000 --host 127.0.0.1 > "$SERVER_OUT_LOG" 2> "$SERVER_ERR_LOG"
