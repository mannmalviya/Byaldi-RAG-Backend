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

ollama serve
echo "OLLAMA Server started"

echo "Starting PDF_Server..."
python PDF_Server.py &

echo "Starting Byaldi_RAG_Pipeline_Server..."
python Byaldi_RAG_Pipeline_Server.py &

# Wait for both processes to complete
wait