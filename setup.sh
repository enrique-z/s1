#!/bin/bash

# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install packages with specific CUDA version
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt

# Create python3 symlink
cd .venv/bin && ln -sf python3.11 python3 && cd ../../

echo "Setup complete! Remember to 'source activate' before running scripts." 