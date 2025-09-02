"""
Easy BERT Fine-tuning with LoRA using Hugging Face
==================================================

This tutorial demonstrates how to fine-tune BERT for text classification using LoRA
(Low-Rank Adaptation) with Hugging Face transformers and PEFT library.

LoRA is a parameter-efficient fine-tuning technique that:
1. Freezes the original model weights
2. Adds small trainable low-rank matrices to specific layers
3. Significantly reduces trainable parameters while maintaining performance

Key Benefits:
- Faster training (fewer parameters to update)
- Lower memory usage
- Easy to switch between different tasks
- Preserves original model knowledge

Author: Xi Rao
Date: 2025-09-02
"""

import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings

warnings.filterwarnings("ignore")


def load_imdb_dataset(sample_size=1000):
    """
    Load and prepare IMDB movie review dataset for sentiment classification

    Args:
        sample_size: Number of samples to use for quick experimentation

    Returns:
        train_dataset, test_dataset: Preprocessed datasets
    """
    print(f"📁 Loading IMDB dataset (sample_size={sample_size})...")

    # Load full IMDB dataset
    dataset = load_dataset("imdb")

    # Create smaller samples for quick experimentation
    if sample_size and sample_size < len(dataset["train"]):
        train_indices = np.random.choice(
            len(dataset["train"]), sample_size, replace=False
        )
        test_indices = np.random.choice(
            len(dataset["test"]), sample_size // 5, replace=False
        )

        train_dataset = dataset["train"].select(train_indices)
        test_dataset = dataset["test"].select(test_indices)
    else:
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

    print(
        f"✅ Loaded {len(train_dataset)} training samples and {len(test_dataset)} test samples"
    )
    return train_dataset, test_dataset


def load_custom_dataset(texts, labels):
    """
    Create a dataset from custom text and label lists

    Args:
        texts: List of text strings
        labels: List of integer labels

    Returns:
        Dataset object compatible with Hugging Face
    """
    print("📁 Creating custom dataset...")
    dataset = Dataset.from_dict({"text": texts, "label": labels})
    print(f"✅ Created dataset with {len(dataset)} samples")
    return dataset


def prepare_tokenizer_and_data(model_name, train_dataset, test_dataset, max_length=128):
    """
    Prepare tokenizer and tokenize datasets

    Args:
        model_name: Name of the pre-trained model (e.g., 'bert-base-uncased')
        train_dataset, test_dataset: Raw datasets with 'text' and 'label' columns
        max_length: Maximum sequence length for tokenization

    Returns:
        tokenizer, tokenized_train, tokenized_test
    """
    print(f"🔤 Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        """Tokenize text data"""
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,  # We'll use dynamic padding in data collator
            max_length=max_length,
        )

    print("🔤 Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    print("✅ Tokenization complete!")
    return tokenizer, tokenized_train, tokenized_test


def setup_model_with_lora(model_name, num_labels, lora_config=None):
    """
    Load pre-trained model and apply LoRA configuration

    Args:
        model_name: Name of the pre-trained model
        num_labels: Number of classification classes
        lora_config: LoraConfig object (if None, uses default)

    Returns:
        PEFT model with LoRA applied
    """
    print(f"🤖 Loading base model: {model_name}...")

    # Load the base model with optimized settings for M3 chips
    if torch.backends.mps.is_available():
        # Optimized for Apple Silicon M3
        torch_dtype = torch.bfloat16
        device_map = None  # Let MPS handle device placement
    elif torch.cuda.is_available():
        torch_dtype = torch.float16
        device_map = "auto"
    else:
        torch_dtype = torch.float32
        device_map = None

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    # Default LoRA configuration if none provided
    if lora_config is None:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,  # Sequence classification
            r=16,  # Rank - size of adaptation
            lora_alpha=32,  # Alpha - scaling parameter
            lora_dropout=0.1,  # Dropout for regularization
            target_modules=["query", "value", "key", "dense"],  # Which modules to adapt
            bias="none",  # Don't adapt bias terms
        )

    print("🔧 Applying LoRA configuration...")
    print(f"   - Rank (r): {lora_config.r}")
    print(f"   - Alpha: {lora_config.lora_alpha}")
    print(f"   - Target modules: {lora_config.target_modules}")
    print(f"   - Dropout: {lora_config.lora_dropout}")

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)

    # Print trainable parameters info
    model.print_trainable_parameters()

    return model


def compute_metrics(eval_pred):
    """
    Compute accuracy, precision, recall, and F1 score
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )

    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}


def setup_training_arguments(
    output_dir="./lora_bert_results", num_epochs=3, batch_size=16
):
    """
    Configure training arguments

    Args:
        output_dir: Directory to save model and logs
        num_epochs: Number of training epochs
        batch_size: Training batch size

    Returns:
        TrainingArguments object
    """
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        eval_strategy="epoch",  # Updated parameter name
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=None,  # Disable wandb/tensorboard logging
        push_to_hub=False,
        # Use bf16 for M3 chips instead of fp16
        bf16=torch.backends.mps.is_available(),  # Better for Apple Silicon
        fp16=False,  # Disable fp16 on M3
    )


def train_bert_with_lora(
    model_name="bert-base-uncased",
    dataset_name="imdb",
    custom_texts=None,
    custom_labels=None,
    sample_size=1000,
    lora_r=16,
    lora_alpha=32,
    num_epochs=3,
    batch_size=16,
    max_length=128,
    output_dir="./lora_bert_results",
):
    """
    Complete BERT fine-tuning pipeline with LoRA

    Args:
        model_name: Pre-trained model to use
        dataset_name: Dataset to load ('imdb' or 'custom')
        custom_texts: List of texts for custom dataset
        custom_labels: List of labels for custom dataset
        sample_size: Number of samples to use (for quick experimentation)
        lora_r: LoRA rank parameter
        lora_alpha: LoRA alpha parameter
        num_epochs: Number of training epochs
        batch_size: Training batch size
        max_length: Maximum sequence length
        output_dir: Directory to save results

    Returns:
        trainer: Trained model trainer object
    """

    print("🚀 Starting BERT Fine-tuning with LoRA")
    print("=" * 50)

    if dataset_name == "imdb":
        train_dataset, test_dataset = load_imdb_dataset(sample_size)
        num_labels = 2  # Positive/Negative sentiment
    elif dataset_name == "custom" and custom_texts and custom_labels:
        # Split custom data into train/test
        split_idx = int(0.8 * len(custom_texts))
        train_dataset = load_custom_dataset(
            custom_texts[:split_idx], custom_labels[:split_idx]
        )
        test_dataset = load_custom_dataset(
            custom_texts[split_idx:], custom_labels[split_idx:]
        )
        num_labels = len(set(custom_labels))
    else:
        raise ValueError("Invalid dataset configuration")

    tokenizer, tokenized_train, tokenized_test = prepare_tokenizer_and_data(
        model_name, train_dataset, test_dataset, max_length
    )

    # Setup model with LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=["query", "value", "key", "dense"],
        bias="none",
    )

    model = setup_model_with_lora(model_name, num_labels, lora_config)

    # Setup training
    training_args = setup_training_arguments(output_dir, num_epochs, batch_size)

    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train the model
    print("\n🏋️ Starting training...")
    trainer.train()

    # Evaluate
    print("\n📊 Evaluating model...")
    eval_results = trainer.evaluate()
    print("\nFinal Results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")

    # Save the model
    print(f"\n💾 Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    return trainer


def load_trained_model(model_path, model_name="bert-base-uncased"):
    """
    Load a trained LoRA model for inference

    Args:
        model_path: Path to saved model directory
        model_name: Original base model name

    Returns:
        model, tokenizer: Loaded model and tokenizer
    """
    print(f"📥 Loading trained model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load base model
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Load LoRA weights
    from peft import PeftModel

    model = PeftModel.from_pretrained(base_model, model_path)

    print("✅ Model loaded successfully!")
    return model, tokenizer


def predict_text(model, tokenizer, texts, device="auto"):
    """
    Make predictions on new text data

    Args:
        model: Trained LoRA model
        tokenizer: Associated tokenizer
        texts: List of texts to classify
        device: Device to run inference on

    Returns:
        predictions: List of predicted class probabilities
    """
    if device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")  # Apple Silicon M3
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    model.to(device)
    model.eval()

    predictions = []

    with torch.no_grad():
        for text in texts:
            # Tokenize
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, padding=True, max_length=128
            ).to(device)

            # Predict
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            predictions.append(probs.cpu().numpy()[0])

    return predictions


def demo_simple_example():
    """
    Simple demo with a tiny custom dataset
    """
    print("\n🎯 Running Simple Demo")
    print("=" * 30)

    # Create a tiny custom dataset
    custom_texts = [
        "I love this movie! It's amazing!",
        "This film is terrible and boring.",
        "What a wonderful story and great acting.",
        "I hate this movie. Waste of time.",
        "Absolutely fantastic! Highly recommend.",
        "Very disappointing and poorly made.",
        "Great cinematography and excellent plot.",
        "Boring and confusing storyline.",
    ]
    custom_labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative

    # Train model
    trainer = train_bert_with_lora(
        model_name="bert-base-uncased",
        dataset_name="custom",
        custom_texts=custom_texts,
        custom_labels=custom_labels,
        lora_r=8,  # Smaller rank for tiny dataset
        lora_alpha=16,
        num_epochs=5,  # More epochs for tiny dataset
        batch_size=4,
        output_dir="./demo_results",
    )

    # Test inference
    print("\n🧪 Testing inference...")
    test_texts = ["This movie is absolutely wonderful!", "I really dislike this film."]

    # Load model for inference (demonstrates save/load cycle)
    model, tokenizer = load_trained_model("./demo_results")
    predictions = predict_text(model, tokenizer, test_texts)

    for text, pred in zip(test_texts, predictions):
        sentiment = "Positive" if pred[1] > pred[0] else "Negative"
        confidence = max(pred) * 100
        print(f"Text: '{text}'")
        print(f"Prediction: {sentiment} (confidence: {confidence:.1f}%)")
        print()


def demo_imdb_example():
    """
    Demo with IMDB dataset (larger, more realistic)
    """
    print("\n🎬 Running IMDB Demo")
    print("=" * 25)

    trainer = train_bert_with_lora(
        model_name="bert-base-uncased",
        dataset_name="imdb",
        sample_size=2000,  # Use 2000 samples for faster training
        lora_r=16,
        lora_alpha=32,
        num_epochs=3,
        batch_size=16,
        output_dir="./imdb_results",
    )

    return trainer


def advanced_lora_configs():
    """
    Examples of different LoRA configurations for different scenarios
    """
    print("\n🔧 Advanced LoRA Configuration Examples")
    print("=" * 45)

    configs = {
        "lightweight": LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=4,  # Very small rank
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=["query", "value"],  # Only attention
            bias="none",
        ),
        "balanced": LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "value", "key", "dense"],
            bias="none",
        ),
        "comprehensive": LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=64,  # Large rank
            lora_alpha=128,
            lora_dropout=0.1,
            target_modules=[  # All linear layers
                "query",
                "value",
                "key",
                "dense",
                "intermediate.dense",
                "output.dense",
            ],
            bias="lora_only",  # Also adapt bias terms
        ),
    }

    for name, config in configs.items():
        print(f"\n{name.upper()} Configuration:")
        print(f"  - Rank: {config.r}")
        print(f"  - Alpha: {config.lora_alpha}")
        print(f"  - Target modules: {len(config.target_modules)} modules")
        print(f"  - Use case: {get_use_case(name)}")


def get_use_case(config_name):
    """Get use case description for configuration"""
    use_cases = {
        "lightweight": "Quick experimentation, limited resources",
        "balanced": "Most general-purpose applications",
        "comprehensive": "Maximum adaptation capability, abundant resources",
    }
    return use_cases[config_name]


if __name__ == "__main__":
    print("🎓 BERT Fine-tuning started with LoRA")
    print("=====================================")

    # Option 1: Simple demo with custom tiny dataset (recommended for learning)
    # demo_simple_example()

    # Option 2: IMDB demo (more realistic but takes longer)
    demo_imdb_example()

    # Option 3: Show advanced configurations
    # advanced_lora_configs()

    print("\n🎉 LoRA BERT training completed!")
