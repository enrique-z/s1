# Quick Deployment Guide for TensorDock

## First Time Setup

1. Create TensorDock instance:
   - Select A100 GPU (80GB)
   - Ubuntu 22.04 LTS
   - At least 200GB storage

2. Clone your fork:
```bash
git clone https://github.com/enrique-z/s1.git
cd s1
```

3. Run setup script:
```bash
chmod +x setup.sh
./setup.sh
source activate
```

4. Model Setup (125GB) - Recommended to do both:

   Step 1 - Create TensorDock Volume (Primary Method):
   ```bash
   # Create directory structure
   mkdir -p ~/.cache/huggingface/models/simplescaling/
   cd ~/.cache/huggingface/models/simplescaling/

   # Download model
   python3 -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
   model = AutoModelForCausalLM.from_pretrained('simplescaling/s1-32B', torch_dtype='auto', device_map='auto'); \
   tokenizer = AutoTokenizer.from_pretrained('simplescaling/s1-32B')"

   # Create TensorDock volume from this directory
   # Name it something like 's1-32b-cache'
   ```

   Step 2 - Backup to Hugging Face Cache (Secondary Method):
   ```bash
   # This happens automatically when you run any script
   # Just run a quick test to verify:
   cd ~/s1
   python test_model.py
   ```

## Subsequent Deployments

### On TensorDock (Fastest):
1. Create instance with:
   - A100 GPU (80GB)
   - Ubuntu 22.04 LTS
   - Mount your 's1-32b-cache' volume to ~/.cache/huggingface/

2. Quick setup:
```bash
git clone https://github.com/enrique-z/s1.git
cd s1
./setup.sh
source activate
python test_simple.py  # Should load instantly
```

### On Other Platforms:
1. Create instance with:
   - A100 GPU (80GB or similar)
   - Ubuntu 22.04 LTS

2. Standard setup:
```bash
git clone https://github.com/enrique-z/s1.git
cd s1
./setup.sh
source activate
python test_simple.py  # Will download model first time
```

## Available Scripts

- `test_simple.py`: Interactive command-line chat (recommended)
- `web_ui.py`: Web interface on port 7860
- `test_model.py`: Basic model test
- `test_model_deep.py`: Complex reasoning test

## Troubleshooting

1. CUDA/Torch issues:
   ```bash
   pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
   ```

2. Python version issues:
   ```bash
   # Already handled by setup.sh
   cd .venv/bin && ln -sf python3.11 python3
   ```

3. Model loading errors:
   ```bash
   # Clear cache and retry
   rm -rf ~/.cache/huggingface/models/simplescaling/s1-32B
   # Then run any script to redownload
   ``` 