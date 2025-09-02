# MiniNeuroLab
A sandbox for training, tweaking and playing with some mini LMs and mixing modalities

## BERT Fine-tuning with LoRA Tutorial

This repository includes a comprehensive tutorial for fine-tuning BERT models using LoRA (Low-Rank Adaptation) with Hugging Face libraries.

### Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the tutorial:**
```bash
cd lora/
python lora_bert_hf.py
```

### What You'll Learn

- **LoRA Fundamentals**: Understanding rank, alpha, and target modules
- **Dataset Preparation**: Working with IMDB, custom datasets, and any classification data
- **Model Setup**: Applying LoRA to pre-trained BERT models
- **Training**: Complete training pipeline with evaluation metrics
- **Inference**: Loading trained models and making predictions
- **Best Practices**: Different configurations for different scenarios

### Tutorial Features

- 🎓 **Educational Focus**: Clear explanations of every concept
- 🚀 **Ready-to-Run**: Complete working examples
- 🔧 **Configurable**: Easy to adapt for your own datasets
- 📊 **Comprehensive**: Includes evaluation, saving, and loading
- ⚡ **Efficient**: Uses LoRA for parameter-efficient fine-tuning

### Examples Included

1. **Simple Demo**: Tiny custom dataset for quick learning
2. **IMDB Demo**: Realistic movie review sentiment classification
3. **Advanced Configs**: Different LoRA setups for various use cases

### Key Benefits of LoRA

- **Faster Training**: Only 0.1-3% of parameters are trainable
- **Lower Memory**: Significant reduction in GPU memory usage
- **Task Switching**: Easy to swap between different fine-tuned models
- **Preserved Knowledge**: Original model knowledge remains intact
