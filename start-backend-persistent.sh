#!/bin/bash

VENV_NAME="ColPali_RAG_Venv"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Try to find the right Python command
if command_exists python3; then
    PYTHON_CMD="python3"
elif command_exists python; then
    PYTHON_CMD="python"
else
    echo "Error: Python not found. Please install Python 3.7 or higher."
    exit 1
fi

# If venv folder exists
if [ -d "$VENV_NAME" ]; then
    echo "Activating existing venv: $VENV_NAME"
    source "$VENV_NAME/bin/activate" # Activate it
else
    echo "Creating new venv: $VENV_NAME"
    $PYTHON_CMD -m venv "$VENV_NAME"
    
    # Check if venv was created successfully
    if [ ! -d "$VENV_NAME" ]; then
        echo "Error: Failed to create virtual environment."
        exit 1
    fi
    
    source "$VENV_NAME/bin/activate" # Activate it
    
    # Check if activation was successful
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "Error: Failed to activate virtual environment."
        exit 1
    fi
    
    echo "Upgrading pip and installing requirements..."
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
fi

echo "Virtual environment activated: $VIRTUAL_ENV"
echo "Python path: $(which python)"
echo "Pip path: $(which pip)"




# Log files
OLLAMA_LOG="ollama_server.log"
PDF_LOG="pdf_server.log"
BYALDI_LOG="byaldi_server.log"

# Maximum log size in bytes (e.g., 10 MB)
MAXSIZE=$((10 * 1024 * 1024))

# Function to truncate logs if they exceed MAXSIZE
truncate_log() {
    local logfile=$1
    if [ -f "$logfile" ] && [ $(stat -c%s "$logfile") -gt $MAXSIZE ]; then
        > "$logfile"
    fi
}

# Truncate logs if needed
truncate_log "$OLLAMA_LOG"
truncate_log "$PDF_LOG"
truncate_log "$BYALDI_LOG"

# Cleanup function on Ctrl+C
cleanup() {
    echo "Stopping servers..."
    kill $OLLAMA_PID $PDF_PID $BYALDI_PID 2>/dev/null
    exit 0
}

# Trap Ctrl+C
trap cleanup SIGINT

# Start Ollama server
nohup ollama serve >> "$OLLAMA_LOG" 2>&1 &
OLLAMA_PID=$!
echo "OLLAMA server started with PID $OLLAMA_PID, logging to $OLLAMA_LOG"

# Start PDF server
nohup python PDF_Server.py >> "$PDF_LOG" 2>&1 &
PDF_PID=$!
echo "PDF server started with PID $PDF_PID, logging to $PDF_LOG"

# Start Byaldi RAG server
nohup python Byaldi_RAG_Pipeline_Server.py >> "$BYALDI_LOG" 2>&1 &
BYALDI_PID=$!
echo "Byaldi RAG server started with PID $BYALDI_PID, logging to $BYALDI_LOG"

# Wait for all background processes
wait
