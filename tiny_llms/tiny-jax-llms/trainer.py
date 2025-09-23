import jax
import jax.numpy as jnp
from jax import random
from dataclasses import dataclass
import requests
from tqdm import tqdm

from models.model_lm import GPTModel, GPTConfig
from flax.training import train_state
import optax
from jax import random, jit
import tiktoken


@dataclass
class TrainingConfig:
    batch_size: int = 8
    num_epochs: int = 3
    learning_rate: float = 1e-3
    eval_every: int = 50


def create_train_state(rng, config: GPTConfig):
    """Create initial training state."""
    model = GPTModel(config)
    
    # Initialize parameters
    dummy_input = jnp.ones((1, config.context_length), dtype=jnp.int32)
    params = model.init(rng, dummy_input, deterministic=True)
    
    # Create optimizer
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=100,
        decay_steps=1000,
        end_value=0.1 * config.learning_rate,
    )
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=schedule,
            weight_decay=config.weight_decay,
        ),
    )
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )


def compute_loss(params, apply_fn, input_ids, target_ids, rng, deterministic=True):
    """Compute cross-entropy loss."""
    logits = apply_fn(params, input_ids, deterministic=deterministic, rngs={'dropout': rng} if not deterministic else None)
    
    # Flatten for loss computation
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = target_ids.reshape(-1)
    
    loss = optax.softmax_cross_entropy_with_integer_labels(logits_flat, targets_flat).mean()
    
    preds = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(preds == target_ids)
    perplexity = jnp.exp(loss)
    
    return loss, {'accuracy': accuracy, 'perplexity': perplexity}


@jit
def train_step(state, input_ids, target_ids, rng):
    """Single training step."""
    dropout_rng = random.fold_in(rng, state.step)
    
    def loss_fn(params):
        return compute_loss(params, state.apply_fn, input_ids, target_ids, dropout_rng, deterministic=False)
    
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads)
    
    return new_state, loss, metrics


@jit
def eval_step(state, input_ids, target_ids):
    """Single evaluation step."""
    loss, metrics = compute_loss(state.params, state.apply_fn, input_ids, target_ids, None, deterministic=True)
    return loss, metrics


def generate_text(state, tokenizer, start_context: str, max_new_tokens: int = 50, temperature: float = 1.0, rng_key=None):
    """Generate text using the trained model."""
    if rng_key is None:
        rng_key = random.PRNGKey(0)
    
    encoded = tokenizer.encode(start_context)
    context_size = state.params['params']['Embed_1']['embedding'].shape[0]  # pos_embed size
    
    # Convert to JAX array
    tokens = jnp.array(encoded, dtype=jnp.int32).reshape(1, -1)

    for _ in range(max_new_tokens):
        # Keep only the last context_size tokens
        current_tokens = tokens[:, -context_size:]
        
        # Get logits
        logits = state.apply_fn(state.params, current_tokens, deterministic=True)
        
        # Get logits for the last token and apply temperature
        next_logits = logits[:, -1, :] / temperature
        
        # Sample next token
        rng_key, sample_key = random.split(rng_key)
        next_token = random.categorical(sample_key, next_logits, axis=-1)
        
        # Append to sequence
        tokens = jnp.concatenate([tokens, next_token.reshape(1, 1)], axis=1)
    
    generated_tokens = tokens.squeeze(0).tolist()
    return tokenizer.decode(generated_tokens)


class SimpleTrainer:
    def __init__(self, config: GPTConfig, train_config: TrainingConfig):
        self.config = config
        self.train_config = train_config
        self.rng = random.PRNGKey(42)
        
    def prepare_data(self, text_data: str, tokenizer):
        token_ids = tokenizer.encode(text_data)
        context_len = self.config.context_length
        
        input_ids = []
        target_ids = []
        
        for i in range(0, len(token_ids) - context_len, context_len):
            if i + context_len + 1 < len(token_ids):
                input_chunk = token_ids[i:i + context_len]
                target_chunk = token_ids[i + 1:i + context_len + 1]
                input_ids.append(jnp.array(input_chunk))
                target_ids.append(jnp.array(target_chunk))
        
        split_idx = int(0.9 * len(input_ids))
        return {
            'train_inputs': input_ids[:split_idx],
            'train_targets': target_ids[:split_idx],
            'val_inputs': input_ids[split_idx:],
            'val_targets': target_ids[split_idx:]
        }
    
    def get_batch(self, inputs, targets, batch_size, rng_key):
        n_samples = len(inputs)
        indices = random.choice(rng_key, n_samples, (batch_size,), replace=False)
        
        batch_inputs = jnp.stack([inputs[i] for i in indices])
        batch_targets = jnp.stack([targets[i] for i in indices])
        return batch_inputs, batch_targets
    
    def evaluate(self, state, val_inputs, val_targets):
        eval_rng = random.PRNGKey(0)
        total_loss = 0.0
        total_acc = 0.0
        n_batches = min(10, len(val_inputs) // self.train_config.batch_size)
        
        for i in range(n_batches):
            eval_rng, batch_rng = random.split(eval_rng)
            val_batch_inputs, val_batch_targets = self.get_batch(
                val_inputs, val_targets, self.train_config.batch_size, batch_rng
            )
            loss, metrics = eval_step(state, val_batch_inputs, val_batch_targets)
            total_loss += loss
            total_acc += metrics['accuracy']
        
        return {
            'loss': float(total_loss / n_batches),
            'accuracy': float(total_acc / n_batches)
        }
    
    def train(self, text_data: str, tokenizer):
        print("Preparing data...")
        data = self.prepare_data(text_data, tokenizer)
        print(f"Train samples: {len(data['train_inputs'])}, Val samples: {len(data['val_inputs'])}")
        
        print("Initializing model...")
        self.rng, init_rng = random.split(self.rng)
        state = create_train_state(init_rng, self.config)
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
        print(f"Parameters: {param_count:,}")
        
        n_train = len(data['train_inputs'])
        steps_per_epoch = n_train // self.train_config.batch_size
        total_steps = steps_per_epoch * self.train_config.num_epochs
        
        print(f"Training for {self.train_config.num_epochs} epochs, {steps_per_epoch} steps each")
        
        step = 0
        best_val_loss = float('inf')
        
        pbar = tqdm(total=total_steps, desc="Training", unit="step")
        
        for epoch in range(self.train_config.num_epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0
            
            for batch_idx in range(steps_per_epoch):
                self.rng, batch_rng, step_rng = random.split(self.rng, 3)
                
                batch_inputs, batch_targets = self.get_batch(
                    data['train_inputs'], data['train_targets'], 
                    self.train_config.batch_size, batch_rng
                )
                
                state, loss, metrics = train_step(state, batch_inputs, batch_targets, step_rng)
                
                epoch_loss += loss
                epoch_acc += metrics['accuracy']
                step += 1
                
                avg_loss = epoch_loss / (batch_idx + 1)
                avg_acc = epoch_acc / (batch_idx + 1)
                
                pbar.set_postfix({
                    'epoch': f"{epoch+1}/{self.train_config.num_epochs}",
                    'loss': f'{avg_loss:.3f}',
                    'acc': f'{avg_acc:.3f}'
                })
                pbar.update(1)
                
                if step % self.train_config.eval_every == 0:
                    val_metrics = self.evaluate(state, data['val_inputs'], data['val_targets'])
                    pbar.write(f"Step {step} - Val Loss: {val_metrics['loss']:.3f}, Val Acc: {val_metrics['accuracy']:.3f}")
                    
                    if val_metrics['loss'] < best_val_loss:
                        best_val_loss = val_metrics['loss']
                        self.best_state = state
        
        pbar.close()
        print(f"Training complete! Best val loss: {best_val_loss:.3f}")
        return self.best_state if hasattr(self, 'best_state') else state


def download_data():
    try:
        response = requests.get(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            timeout=10
        )
        if response.status_code == 200:
            return response.text
    except:
        pass
    
    fallback = """To be or not to be, that is the question.
    Whether 'tis nobler in the mind to suffer the slings and arrows of outrageous fortune,
    or to take arms against a sea of troubles and by opposing end them.
    The quick brown fox jumps over the lazy dog. """ * 1000
    
    return fallback


def main():
    print("JAX devices:", jax.devices())
    
    config = GPTConfig(
        vocab_size=50257,
        context_length=128,
        emb_dim=256,
        n_heads=8,
        n_layers=4,
        learning_rate=1e-3
    )
    
    train_config = TrainingConfig(
        batch_size=8,
        num_epochs=3,
        eval_every=50
    )
    
    print("Loading data...")
    text_data = download_data()
    print(f"Data length: {len(text_data):,} characters")
    
    tokenizer = tiktoken.get_encoding("gpt2")
    trainer = SimpleTrainer(config, train_config)
    
    state = trainer.train(text_data, tokenizer)
    
    print("\nTesting generation...")
    prompts = ["To be or not", "The quick brown"]
    
    for prompt in prompts:
        generated = generate_text(state, tokenizer, prompt, max_new_tokens=20, temperature=0.8)
        print(f"'{prompt}' -> {generated}")
    
    return state, tokenizer


if __name__ == "__main__":
    state, tokenizer = main()