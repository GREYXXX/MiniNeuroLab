# Diffusion Model Training Framework

A professional, modular implementation of text-to-image diffusion models for educational purposes. This framework supports both U-Net and Diffusion Transformer (DiT) architectures with PyTorch Lightning for training.

## Architecture Overview

### Components

1. **Text Encoder** (`models/text_encoder.py`)
   - Transformer-based encoder for text prompts
   - Converts text tokens to semantic embeddings
   - Similar to CLIP's text encoder

2. **VAE** (`models/vae.py`)
   - Encoder: Compresses images to latent space (8x downsampling)
   - Decoder: Reconstructs images from latents
   - Enables efficient diffusion in lower-dimensional space

3. **Denoising Networks**
   - **U-Net** (`models/unet.py`): CNN-based architecture with attention
   - **DiT** (`models/dit.py`): Transformer-based architecture (Vision Transformer style)

4. **Diffusion Scheduler** (`scheduler.py`)
   - DDPM noise scheduling
   - Forward and reverse diffusion processes
   - Supports linear and cosine schedules

5. **Training Module** (`train.py`)
   - PyTorch Lightning integration
   - Automatic mixed precision training
   - Checkpointing and logging

## Project Structure

```
diffusion/
├── models/
│   ├── __init__.py
│   ├── text_encoder.py      # Text encoder
│   ├── vae.py               # VAE encoder/decoder
│   ├── unet.py              # U-Net denoising model
│   ├── dit.py               # Diffusion Transformer
│   └── diffusion_model.py   # Main model combining all components
├── scheduler.py              # DDPM scheduler
├── data.py                  # Dataset classes
├── train.py                 # PyTorch Lightning training
├── main.py                  # Main training script
└── README.md                # This file
```

## Quick Start

### Installation

```bash
pip install torch pytorch-lightning torchvision
```

### Training with U-Net

```bash
python main.py \
    --denoising_type unet \
    --epochs 10 \
    --batch_size 4 \
    --learning_rate 1e-4
```

### Training with DiT

```bash
python main.py \
    --denoising_type dit \
    --epochs 10 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --dit_depth 12 \
    --dit_num_heads 12
```

## Key Features

### 1. Modular Architecture
- Clean separation of concerns
- Easy to modify individual components
- Professional code structure

### 2. Dual Denoising Networks
- **U-Net**: Standard CNN-based architecture (Stable Diffusion style)
- **DiT**: Modern transformer-based architecture (scalable and efficient)

### 3. PyTorch Lightning Integration
- Automatic mixed precision training
- Gradient accumulation
- Distributed training support
- Built-in checkpointing and logging
- Learning rate scheduling

### 4. Educational Focus
- Clear comments explaining each component
- Well-documented code
- Easy to understand and modify

## Configuration Options

### Model Configuration

- `--denoising_type`: Choose "unet" or "dit"
- `--vocab_size`: Vocabulary size for text encoder
- `--text_emb_dim`: Text embedding dimension
- `--latent_dim`: Latent space dimension

### Training Configuration

- `--epochs`: Number of training epochs
- `--batch_size`: Batch size
- `--learning_rate`: Learning rate
- `--weight_decay`: Weight decay for optimizer
- `--num_timesteps`: Number of diffusion timesteps

### U-Net Specific

- `--unet_base_channels`: Base number of channels

### DiT Specific

- `--dit_hidden_size`: Hidden dimension
- `--dit_depth`: Number of transformer layers
- `--dit_num_heads`: Number of attention heads
- `--dit_patch_size`: Patch size for patch embedding

## Usage Examples

### Basic Training

```python
from models import LatentDiffusionModel
from scheduler import DDPMScheduler
from data import create_dummy_dataset
from train import train_diffusion_model

# Create model
model = LatentDiffusionModel(
    vocab_size=10000,
    denoising_type="unet",  # or "dit"
)

# Create scheduler
scheduler = DDPMScheduler(num_timesteps=1000)

# Create dataset
train_dataset = create_dummy_dataset(num_samples=100)

# Train
trainer = train_diffusion_model(
    model=model,
    scheduler=scheduler,
    train_dataset=train_dataset,
)
```

### Generation

```python
# After training, generate images
model.eval()
with torch.no_grad():
    text_tokens = torch.randint(0, 10000, (1, 77))
    images = model.generate(
        text_tokens=text_tokens,
        scheduler=scheduler,
        num_inference_steps=50,
        guidance_scale=7.5,
        image_size=(256, 256),
    )
```

## Key Concepts Explained

### 1. Latent Diffusion
- Images are encoded to latent space (8x smaller)
- Diffusion happens in latent space (more efficient)
- Latents are decoded back to images

### 2. Text Conditioning
- Text prompts are encoded to embeddings
- Cross-attention allows image features to attend to text
- Enables text-to-image generation

### 3. Diffusion Process
- Forward: Gradually add noise to images
- Reverse: Gradually remove noise (generation)
- Model learns to predict noise at each timestep

### 4. U-Net vs DiT
- **U-Net**: CNN-based, good for spatial features
- **DiT**: Transformer-based, better scalability, patch-based processing

## Training Tips

1. **Start Small**: Use small models and datasets for initial experiments
2. **Mixed Precision**: Use `--precision 16-mixed` for faster training
3. **Gradient Clipping**: Helps with training stability
4. **Learning Rate**: Start with 1e-4 and adjust based on loss curves
5. **Batch Size**: Adjust based on available memory

## Extending the Framework

### Adding Custom Datasets

Modify `data.py` to add your dataset:

```python
class CustomDataset(TextImageDataset):
    def __init__(self, ...):
        # Your implementation
        pass
```

### Custom Denoising Networks

Add new architectures in `models/` and update `diffusion_model.py`:

```python
# In diffusion_model.py
elif denoising_type == "your_architecture":
    self.denoising_net = YourArchitecture(...)
```

### Custom Schedules

Extend `scheduler.py` to add new noise schedules:

```python
class CustomScheduler(DDPMScheduler):
    def __init__(self, ...):
        # Your custom schedule
        pass
```

## References

- **DDPM**: Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
- **Stable Diffusion**: Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" (2022)
- **DiT**: Peebles & Xie, "Scalable Diffusion Models with Transformers" (2023)

## License

This code is for educational purposes. Feel free to modify and experiment!