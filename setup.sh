#!/bin/bash

# Specify the name of your virtual environment
venv_name=".venv"

# Check if the virtual environment already exists
if [ ! -d "$venv_name" ]; then
    # Create a new virtual environment
    python3 -m venv "$venv_name"
    echo "Created virtual environment: $venv_name"
fi

# Activate the virtual environment
source "$venv_name/bin/activate"

# Install pip upgrade
pip3 install --upgrade pip

# Install project dependencies using pip
pip3 install -r requirements.txt

echo "Virtual environment setup complete"
