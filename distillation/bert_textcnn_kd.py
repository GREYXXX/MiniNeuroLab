import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import time
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings("ignore")

torch.manual_seed(42)
np.random.seed(42)


class TextCNN(nn.Module):
    """
    TextCNN model for text classification

    Architecture:
    1. Embedding layer (word → vectors)
    2. Multiple CNN layers with different kernel sizes
    3. Global max pooling
    4. Fully connected layers
    5. Classification head
    """

    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        num_filters=100,
        filter_sizes=[3, 4, 5],
        num_classes=2,
        dropout=0.3,
    ):
        super(TextCNN, self).__init__()

        self.embed_dim = embed_dim
        self.num_filters = num_filters
        self.num_classes = num_classes

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, kernel_size=k) for k in filter_sizes]
        )

        # Feature extraction layers for knowledge distillation
        self.feature_dim = num_filters * len(filter_sizes)
        self.feature_layer = nn.Linear(self.feature_dim, 768)  # Match BERT hidden size

        # Classification layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.feature_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.uniform_(module.weight, -0.1, 0.1)

    def forward(self, input_ids, return_features=False):
        # Embedding: [batch_size, seq_len] → [batch_size, seq_len, embed_dim]
        embedded = self.embedding(input_ids)

        # CNN expects [batch_size, embed_dim, seq_len]
        embedded = embedded.transpose(1, 2)

        # Apply multiple CNN filters
        conv_outputs = []
        for conv in self.convs:
            # Conv1d: [batch_size, embed_dim, seq_len] → [batch_size, num_filters, new_len]
            conv_out = F.relu(conv(embedded))
            # Global max pooling: [batch_size, num_filters, new_len] → [batch_size, num_filters]
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)

        # Concatenate all filter outputs: [batch_size, num_filters * len(filter_sizes)]
        features = torch.cat(conv_outputs, dim=1)

        # Extract features for knowledge distillation
        if return_features:
            # Project to BERT feature space
            distill_features = self.feature_layer(features)

        # Classification head
        x = self.dropout(features)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)

        if return_features:
            return logits, distill_features
        return logits

    def count_parameters(self):
        """Count total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BERTTeacher(nn.Module):
    """
    BERT teacher model wrapper for knowledge distillation
    """

    def __init__(self, model_name="bert-base-uncased", num_classes=2):
        super(BERTTeacher, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes
        )

        # Freeze all parameters (teacher is pre-trained)
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()  # Always in eval mode

    def forward(self, input_ids, attention_mask=None, return_features=False):
        # Teacher is frozen
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=return_features,
            )

            logits = outputs.logits

            if return_features:
                # Use [CLS] token representation from last hidden layer
                hidden_states = outputs.hidden_states[
                    -1
                ]  # [batch_size, seq_len, hidden_size]
                cls_features = hidden_states[:, 0, :]  # [batch_size, hidden_size]
                return logits, cls_features

            return logits

    def count_parameters(self):
        """Count total number of parameters"""
        return sum(p.numel() for p in self.model.parameters())


class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation:
    1. Distillation Loss: KL divergence between teacher and student predictions
    2. Task Loss: Cross-entropy with ground truth labels
    3. Feature Loss: MSE between teacher and student features
    """

    def __init__(self, temperature=3.0, alpha=0.7, beta=0.2):
        super(DistillationLoss, self).__init__()

        self.temperature = temperature  # Temperature for softening probabilities
        self.alpha = alpha  # Weight for distillation loss
        self.beta = beta  # Weight for feature matching loss
        # gamma = 1 - alpha - beta for task loss

        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        student_logits,
        teacher_logits,
        labels,
        student_features=None,
        teacher_features=None,
    ):
        # 1. Distillation Loss (KL divergence between soft predictions)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        distill_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature**2)

        # 2. Task Loss (standard cross-entropy with labels)
        task_loss = self.ce_loss(student_logits, labels)

        # 3. Feature Matching Loss (if features provided)
        feature_loss = 0.0
        if student_features is not None and teacher_features is not None:
            feature_loss = self.mse_loss(student_features, teacher_features)

        # Combined loss
        total_loss = (
            self.alpha * distill_loss
            + (1 - self.alpha - self.beta) * task_loss
            + self.beta * feature_loss
        )

        loss_dict = {
            "total_loss": total_loss.item(),
            "distill_loss": distill_loss.item(),
            "task_loss": task_loss.item(),
            "feature_loss": (
                feature_loss if isinstance(feature_loss, float) else feature_loss.item()
            ),
        }

        return total_loss, loss_dict


class TextDataset(Dataset):
    """
    Custom dataset for text classification with both BERT and TextCNN tokenization
    """

    def __init__(self, texts, labels, bert_tokenizer, vocab_dict, max_length=128):
        self.texts = texts
        self.labels = labels
        self.bert_tokenizer = bert_tokenizer
        self.vocab_dict = vocab_dict
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # BERT tokenization
        bert_encoding = self.bert_tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # TextCNN tokenization (simple word-level)
        words = text.lower().split()[: self.max_length]
        textcnn_ids = [
            self.vocab_dict.get(word, self.vocab_dict["<UNK>"]) for word in words
        ]

        # Pad to max_length
        if len(textcnn_ids) < self.max_length:
            textcnn_ids.extend(
                [self.vocab_dict["<PAD>"]] * (self.max_length - len(textcnn_ids))
            )

        return {
            "bert_input_ids": bert_encoding["input_ids"].squeeze(),
            "bert_attention_mask": bert_encoding["attention_mask"].squeeze(),
            "textcnn_input_ids": torch.tensor(textcnn_ids, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def build_vocab(texts, max_vocab_size=10000):
    """
    Build vocabulary for TextCNN from text data

    Args:
        texts: List of text strings
        max_vocab_size: Maximum vocabulary size

    Returns:
        vocab_dict: Dictionary mapping words to indices
    """
    word_count = {}

    # Count word frequencies
    for text in texts:
        words = str(text).lower().split()
        for word in words:
            word_count[word] = word_count.get(word, 0) + 1

    # Sort by frequency and take top words
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    top_words = sorted_words[: max_vocab_size - 3]  # Reserve space for special tokens

    # Build vocabulary dictionary
    vocab_dict = {"<PAD>": 0, "<UNK>": 1, "<START>": 2}

    for word, _ in top_words:
        vocab_dict[word] = len(vocab_dict)

    return vocab_dict


def load_dataset_for_distillation(dataset_name="imdb", sample_size=5000):
    """
    Load and prepare dataset for knowledge distillation

    Args:
        dataset_name: Name of dataset to load
        sample_size: Number of samples to use (for quick experimentation)

    Returns:
        train_texts, train_labels, test_texts, test_labels, vocab_dict
    """
    print(f"📁 Loading {dataset_name} dataset...")

    if dataset_name == "imdb":
        # Show progress for dataset loading
        with tqdm(total=1, desc="Loading IMDB dataset") as pbar:
            dataset = load_dataset("imdb")
            pbar.update(1)

        # Sample data for faster experimentation
        if sample_size:
            print(f"🎲 Sampling {sample_size} training samples...")
            train_indices = np.random.choice(
                len(dataset["train"]),
                min(sample_size, len(dataset["train"])),
                replace=False,
            )
            test_indices = np.random.choice(
                len(dataset["test"]),
                min(sample_size // 5, len(dataset["test"])),
                replace=False,
            )

            train_data = dataset["train"].select(train_indices)
            test_data = dataset["test"].select(test_indices)
        else:
            train_data = dataset["train"]
            test_data = dataset["test"]

        train_texts = train_data["text"]
        train_labels = train_data["label"]
        test_texts = test_data["text"]
        test_labels = test_data["label"]

        num_classes = 2

    elif dataset_name == "ag_news":
        dataset = load_dataset("ag_news")

        if sample_size:
            train_indices = np.random.choice(
                len(dataset["train"]),
                min(sample_size, len(dataset["train"])),
                replace=False,
            )
            test_indices = np.random.choice(
                len(dataset["test"]),
                min(sample_size // 5, len(dataset["test"])),
                replace=False,
            )

            train_data = dataset["train"].select(train_indices)
            test_data = dataset["test"].select(test_indices)
        else:
            train_data = dataset["train"]
            test_data = dataset["test"]

        train_texts = train_data["text"]
        train_labels = train_data["label"]
        test_texts = test_data["text"]
        test_labels = test_data["label"]

        num_classes = 4

    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    # Build vocabulary for TextCNN
    print("🔤 Building vocabulary for TextCNN...")
    vocab_dict = build_vocab(train_texts)

    print(f"✅ Dataset loaded:")
    print(f"   - Training samples: {len(train_texts)}")
    print(f"   - Test samples: {len(test_texts)}")
    print(f"   - Number of classes: {num_classes}")
    print(f"   - Vocabulary size: {len(vocab_dict)}")

    return train_texts, train_labels, test_texts, test_labels, vocab_dict, num_classes


class KnowledgeDistillationTrainer:
    """
    Trainer class for knowledge distillation
    """

    def __init__(self, teacher, student, distill_loss, device="auto"):
        self.teacher = teacher
        self.student = student
        self.distill_loss = distill_loss

        # Device setup
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"🖥️  Using device: {self.device}")

        # Move models to device
        self.teacher.to(self.device)
        self.student.to(self.device)

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "loss_components": [],
        }

    def train_epoch(self, train_loader, optimizer, epoch):
        """Train for one epoch"""
        self.student.train()
        self.teacher.eval()

        total_loss = 0.0
        total_samples = 0
        loss_components = {"distill_loss": 0, "task_loss": 0, "feature_loss": 0}

        # Create progress bar for training batches
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            bert_input_ids = batch["bert_input_ids"].to(self.device)
            bert_attention_mask = batch["bert_attention_mask"].to(self.device)
            textcnn_input_ids = batch["textcnn_input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Teacher forward pass (frozen)
            teacher_logits, teacher_features = self.teacher(
                bert_input_ids, bert_attention_mask, return_features=True
            )

            # Student forward pass
            student_logits, student_features = self.student(
                textcnn_input_ids, return_features=True
            )

            # Calculate distillation loss
            loss, loss_dict = self.distill_loss(
                student_logits,
                teacher_logits,
                labels,
                student_features,
                teacher_features,
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update statistics
            total_loss += loss.item()
            total_samples += len(labels)

            for key in loss_components:
                loss_components[key] += loss_dict[key]

            # Update progress bar with current loss
            current_avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix(
                {
                    "Loss": f"{current_avg_loss:.4f}",
                    "Distill": f'{loss_dict["distill_loss"]:.4f}',
                    "Task": f'{loss_dict["task_loss"]:.4f}',
                    "Feature": f'{loss_dict["feature_loss"]:.4f}',
                }
            )

        avg_loss = total_loss / len(train_loader)
        for key in loss_components:
            loss_components[key] /= len(train_loader)

        return avg_loss, loss_components

    def evaluate(self, val_loader):
        """Evaluate model performance"""
        self.student.eval()
        self.teacher.eval()

        total_loss = 0.0
        predictions = []
        true_labels = []

        # Create progress bar for evaluation
        pbar = tqdm(val_loader, desc="Evaluating", leave=False)
        with torch.no_grad():
            for batch in pbar:
                # Move batch to device
                bert_input_ids = batch["bert_input_ids"].to(self.device)
                bert_attention_mask = batch["bert_attention_mask"].to(self.device)
                textcnn_input_ids = batch["textcnn_input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward passes
                teacher_logits, teacher_features = self.teacher(
                    bert_input_ids, bert_attention_mask, return_features=True
                )
                student_logits, student_features = self.student(
                    textcnn_input_ids, return_features=True
                )

                # Calculate loss
                loss, _ = self.distill_loss(
                    student_logits,
                    teacher_logits,
                    labels,
                    student_features,
                    teacher_features,
                )

                total_loss += loss.item()

                # Collect predictions
                preds = torch.argmax(student_logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

                # Update progress bar
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(true_labels, predictions)

        return avg_loss, accuracy, predictions, true_labels

    def train(self, train_loader, val_loader, num_epochs=5, learning_rate=1e-3):
        print("🏋️ Starting Knowledge Distillation Training...")
        print(f"Teacher parameters: {self.teacher.count_parameters():,}")
        print(f"Student parameters: {self.student.count_parameters():,}")
        print(
            f"Compression ratio: {self.teacher.count_parameters() / self.student.count_parameters():.1f}x"
        )
        print()

        optimizer = optim.Adam(self.student.parameters(), lr=learning_rate)
        epoch_pbar = tqdm(range(num_epochs), desc="Training Progress", position=0)
        for epoch in epoch_pbar:
            start_time = time.time()
            train_loss, loss_components = self.train_epoch(
                train_loader, optimizer, epoch + 1
            )
            val_loss, val_accuracy, _, _ = self.evaluate(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_accuracy)
            self.history["loss_components"].append(loss_components)

            epoch_time = time.time() - start_time
            # Update epoch progress bar
            epoch_pbar.set_postfix(
                {
                    "Train Loss": f"{train_loss:.4f}",
                    "Val Acc": f"{val_accuracy:.4f}",
                    "Time": f"{epoch_time:.1f}s",
                }
            )

            # Print detailed results
            tqdm.write(f"Epoch {epoch + 1}/{num_epochs}:")
            tqdm.write(f"  Train Loss: {train_loss:.4f}")
            tqdm.write(f"  Val Loss: {val_loss:.4f}")
            tqdm.write(f"  Val Accuracy: {val_accuracy:.4f}")
            tqdm.write(f"  Distill Loss: {loss_components['distill_loss']:.4f}")
            tqdm.write(f"  Task Loss: {loss_components['task_loss']:.4f}")
            tqdm.write(f"  Feature Loss: {loss_components['feature_loss']:.4f}")
            tqdm.write(f"  Time: {epoch_time:.2f}s")
            tqdm.write("-" * 50)

        return self.history


def compare_models(teacher, student, test_loader, device):
    """
    Compare teacher and student model performance

    Args:
        teacher: Teacher model
        student: Student model
        test_loader: Test data loader
        device: Computing device

    Returns:
        comparison_results: Dictionary with comparison metrics
    """
    print("📊 Comparing Teacher vs Student Performance...")

    teacher.eval()
    student.eval()

    teacher_preds = []
    student_preds = []
    true_labels = []

    teacher_time = 0
    student_time = 0

    # Create progress bar for model comparison
    pbar = tqdm(test_loader, desc="Comparing Models", leave=False)

    with torch.no_grad():
        for batch in pbar:
            bert_input_ids = batch["bert_input_ids"].to(device)
            bert_attention_mask = batch["bert_attention_mask"].to(device)
            textcnn_input_ids = batch["textcnn_input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Teacher inference
            start_time = time.time()
            teacher_logits = teacher(bert_input_ids, bert_attention_mask)
            teacher_time += time.time() - start_time

            # Student inference
            start_time = time.time()
            student_logits = student(textcnn_input_ids)
            student_time += time.time() - start_time

            # Collect predictions
            teacher_preds.extend(torch.argmax(teacher_logits, dim=1).cpu().numpy())
            student_preds.extend(torch.argmax(student_logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    teacher_accuracy = accuracy_score(true_labels, teacher_preds)
    student_accuracy = accuracy_score(true_labels, student_preds)

    # Model sizes
    teacher_params = teacher.count_parameters()
    student_params = student.count_parameters()

    results = {
        "teacher_accuracy": teacher_accuracy,
        "student_accuracy": student_accuracy,
        "accuracy_retention": student_accuracy / teacher_accuracy,
        "teacher_params": teacher_params,
        "student_params": student_params,
        "compression_ratio": teacher_params / student_params,
        "teacher_time": teacher_time,
        "student_time": student_time,
        "speedup": teacher_time / student_time,
    }

    print(f"Teacher Accuracy: {teacher_accuracy:.4f}")
    print(f"Student Accuracy: {student_accuracy:.4f}")
    print(f"Accuracy Retention: {results['accuracy_retention']:.2%}")
    print(f"Model Compression: {results['compression_ratio']:.1f}x smaller")
    print(f"Inference Speedup: {results['speedup']:.1f}x faster")

    return results


def plot_training_history(history):
    """Plot training metrics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Accuracy curve
    ax2.plot(epochs, history["val_accuracy"], "g-", label="Val Accuracy")
    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    # Loss components
    distill_losses = [comp["distill_loss"] for comp in history["loss_components"]]
    task_losses = [comp["task_loss"] for comp in history["loss_components"]]
    feature_losses = [comp["feature_loss"] for comp in history["loss_components"]]

    ax3.plot(epochs, distill_losses, "b-", label="Distillation Loss")
    ax3.plot(epochs, task_losses, "r-", label="Task Loss")
    ax3.plot(epochs, feature_losses, "g-", label="Feature Loss")
    ax3.set_title("Loss Components")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")
    ax3.legend()
    ax3.grid(True)

    # Model comparison
    ax4.text(
        0.1,
        0.9,
        "Knowledge Distillation Results",
        transform=ax4.transAxes,
        fontsize=14,
        fontweight="bold",
    )
    ax4.text(
        0.1,
        0.7,
        f'Final Val Accuracy: {history["val_accuracy"][-1]:.4f}',
        transform=ax4.transAxes,
        fontsize=12,
    )
    ax4.text(
        0.1, 0.5, f"Total Epochs: {len(epochs)}", transform=ax4.transAxes, fontsize=12
    )
    ax4.axis("off")

    plt.tight_layout()
    plt.savefig("knowledge_distillation_results.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    config = {
        "dataset": "imdb",  # 'imdb' or 'ag_news'
        "sample_size": 2000,  # Use subset for faster training
        "max_length": 128,
        "batch_size": 32,
        "num_epochs": 5,
        "learning_rate": 1e-3,
        "temperature": 4.0,
        "alpha": 0.7,  # Distillation loss weight
        "beta": 0.2,  # Feature loss weight
    }

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Load dataset
    train_texts, train_labels, test_texts, test_labels, vocab_dict, num_classes = (
        load_dataset_for_distillation(
            dataset_name=config["dataset"], sample_size=config["sample_size"]
        )
    )

    teacher = BERTTeacher(num_classes=num_classes)
    student = TextCNN(
        vocab_size=len(vocab_dict),
        embed_dim=128,
        num_filters=100,
        filter_sizes=[3, 4, 5],
        num_classes=num_classes,
        dropout=0.3,
    )

    print(f"✅ Teacher (BERT): {teacher.count_parameters():,} parameters")
    print(f"✅ Student (TextCNN): {student.count_parameters():,} parameters")
    print(
        f"🗜️  Compression ratio: {teacher.count_parameters() / student.count_parameters():.1f}x"
    )
    print()

    train_dataset = TextDataset(
        train_texts, train_labels, teacher.tokenizer, vocab_dict, config["max_length"]
    )
    test_dataset = TextDataset(
        test_texts, test_labels, teacher.tokenizer, vocab_dict, config["max_length"]
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False
    )
    distill_loss = DistillationLoss(
        temperature=config["temperature"], alpha=config["alpha"], beta=config["beta"]
    )

    # Training
    trainer = KnowledgeDistillationTrainer(teacher, student, distill_loss)
    history = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=config["num_epochs"],
        learning_rate=config["learning_rate"],
    )

    results = compare_models(teacher, student, test_loader, trainer.device)
    plot_training_history(history)
    torch.save(
        {
            "model_state_dict": student.state_dict(),
            "vocab_dict": vocab_dict,
            "config": config,
            "results": results,
            "history": history,
        },
        "textcnn_student_model.pth",
    )

    print("✅ Model saved as 'textcnn_student_model.pth'")

    print("\n🎉 Knowledge Distillation Complete!")
    print(f"🎯 Final Results:")
    print(f"   Student Accuracy: {results['student_accuracy']:.4f}")
    print(f"   Accuracy Retention: {results['accuracy_retention']:.2%}")
    print(f"   Model Size Reduction: {results['compression_ratio']:.1f}x")
    print(f"   Inference Speedup: {results['speedup']:.1f}x")


if __name__ == "__main__":
    main()
