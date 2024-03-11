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

# Install torch==1.12.0+cu113 for training on dgx
pip3 install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Install project dependencies using pip
pip3 install -r remote_requirements.txt

echo "Virtual environment setup complete"
