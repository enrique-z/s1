# Quick Deployment Guide for TensorDock

## Initial Setup

1. Create TensorDock instance:
   - Select A100 GPU (80GB)
   - Ubuntu 22.04 LTS
   - At least 200GB storage
   - Mount HF cache volume if available

2. Clone your forked repository:
```bash
git clone https://github.com/YOUR_USERNAME/s1.git
cd s1
```

3. Run setup script:
```bash
chmod +x setup.sh
./setup.sh
```

4. Activate environment:
```bash
source activate
```

## Using Pre-downloaded Model

If you have a TensorDock volume with the model:
1. Mount the volume during instance creation
2. Link the model files to HF cache:
```bash
mkdir -p ~/.cache/huggingface/
ln -s /path/to/volume/s1-32B ~/.cache/huggingface/
```

## Creating Backup Image

1. Clean unnecessary files:
```bash
rm -rf .venv/lib/python*/site-packages/torch/test
rm -rf .venv/lib/python*/site-packages/*/tests
```

2. Save HF cache if needed:
```bash
mv ~/.cache/huggingface/ ~/hf_cache_backup/
```

3. Create TensorDock image from instance

## Restore from Backup

1. Launch new instance from saved image
2. Restore HF cache if backed up:
```bash
mv ~/hf_cache_backup/ ~/.cache/huggingface/
```

3. Run your scripts! 