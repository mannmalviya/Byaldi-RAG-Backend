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

# Function to handle Ctrl+C
cleanup() {
    echo "Stopping servers..."
    kill $PDF_PID $BYALDI_PID
    exit 0
}

# Trap Ctrl+C (SIGINT) and call cleanup
trap cleanup SIGINT

# Start OLLAMA server (foreground)
ollama serve &
OLLAMA_PID=$!

echo "OLLAMA Server started with PID $OLLAMA_PID"

# Start PDF_Server in background
python PDF_Server.py &
PDF_PID=$!

# Start Byaldi_RAG_Pipeline_Server in background
python Byaldi_RAG_Pipeline_Server.py &
BYALDI_PID=$!

# Wait for all background processes
wait
