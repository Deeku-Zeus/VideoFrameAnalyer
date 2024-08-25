#!/bin/bash

# Define the path to the virtual environment
VENV_PATH="path/to/venv"

# Create the virtual environment
python3 -m venv "$VENV_PATH"

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Install dependencies from requirements.txt
pip install -r requirements.txt

echo "Setup complete. Virtual environment created at $VENV_PATH and dependencies installed."