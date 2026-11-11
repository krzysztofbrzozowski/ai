import pathlib
import keras
import keras_hub
import numpy as np

# --- START LOADING DATASET
PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent

extract_dir = keras.utils.get_file(
    fname="mini-c4",
    origin=(
        "https://hf.co/datasets/mattdangerw/mini-c4/resolve/main/mini-c4.zip"
    ),
    extract=True,
    cache_dir=PROJECT_DIR,
)
extract_dir = pathlib.Path(extract_dir) / "mini-c4"

vocabulary_file = keras.utils.get_file(
    origin="https://hf.co/mattdangerw/spiece/resolve/main/vocabulary.proto",
    cache_dir=PROJECT_DIR,
)
tokenizer = keras_hub.tokenizers.SentencePieceTokenizer(vocabulary_file)

# Example usage of tokenizer
# >>> tokenizer.tokenize("The quick brown fox.")
# array([  450,  4996, 17354,  1701, 29916, 29889], dtype=int32)
# >>> tokenizer.detokenize([450, 4996, 17354, 1701, 29916, 29889])
# "The quick brown fox."

# --- START DATASET PREPARATION
import tensorflow as tf
# Turn on debugging
# tf.data.experimental.enable_debug_mode()

batch_size = 64
sequence_length = 256
suffix = np.array([tokenizer.token_to_id("<|endoftext|>")])

def read_file(filename):
    # Splits the .txt file from disk by physical line endings.
    # Since each full article in shard*.txt occupies one physical line,
    # from this point on: 1 element x in ds = 1 full article:
    #   x_0 = "Beginners BBQ Class...\\nDo you want..."
    #   x_1 = "Discussion in 'Mac OS X Lion...'\\nI've got..."
    ds = tf.data.TextLineDataset(filename)

    # Replaces literal "\\n" characters inside each article string x with real newlines ("\n").
    # IMPORTANT: This does NOT split element x into multiple dataset lines!
    # x is still a single string (one full article), just multiline inside.
    ds = ds.map(lambda x: tf.strings.regex_replace(x, r"\\n", "\n"))

    # Converts the entire text of each article x into a 1D array of token IDs:
    #   x_0 = [18723, 29871, ..., 29889] (188 tokens)
    #   x_1 = [11234, 542, ..., 29889]   (429 tokens)
    ds = ds.map(tokenizer, num_parallel_calls=8)

    # `suffix` is a single-element array `array([0])` (where 0 is the `<|endoftext|>` token ID).
    # For each token array x (i.e., for each article), `tf.concat([x, suffix], -1)`
    # appends 0 to the very end of the vector (axis=-1):
    #   x_0 = [18723, 29871, ..., 29889] + [0] -> [18723, 29871, ..., 29889, 0] (189 tokens)
    #   x_1 = [11234, 542, ..., 29889]   + [0] -> [11234, 542, ..., 29889, 0]   (430 tokens)
    # That is why `<|endoftext|>` lands at the end of each full article
    # (each original line from shard*.txt), rather than after every sentence inside an article.
    return ds.map(lambda x: tf.concat([x, suffix], -1))

files = [str(file) for file in extract_dir.glob("*.txt")]
ds = tf.data.Dataset.from_tensor_slices(files)
# Combines our file shards into a single dataset
ds = ds.interleave(read_file, cycle_length=32, num_parallel_calls=32)
# Windows tokens into even samples of 256 tokens
ds = ds.rebatch(sequence_length + 1, drop_remainder=True)
# Splits labels, offset by one
ds = ds.map(lambda x: (x[:-1], x[1:]))
ds = ds.batch(batch_size).prefetch(8)

# ds.reduce(0, lambda c, _: c + 1)
# Verify the number of batches in ds:
# c = 0               # <- first argument of reduce: initial_state = 0
# for _ in ds:        # <- second argument: lambda c, _: c + 1
#     c = c + 1
# return c            # -> returns a tf.Tensor with the final element count
#
# For a pipeline built with TextLineDataset -> interleave -> rebatch, TensorFlow does not know
# in advance how many lines of text or how many tokens will be produced from the 50 .txt files on disk:
# - len(ds) raises TypeError: The dataset length is unknown,
# - ds.cardinality().numpy() returns -2 (UNKNOWN_CARDINALITY).
#
# Therefore, ds.reduce(0, lambda c, _: c + 1) forces a full pass through the entire data pipeline
# from start to finish in order to count the exact number of batches it produces.
# ---
# When you call ds.reduce(0, lambda c, _: c + 1), the computer must immediately:
# 1. Read all ~3.8 GB of text from disk,
# 2. Tokenize nearly one billion tokens on the CPU,
# 3. Slice them into 256-token windows and group them into 58,746 batches,
# 4. Discard each batch while only incrementing the counter c + 1.

# --- START TRAINING AND VALIDATION DATASET
num_batches = 58746
num_val_batches = 500
num_train_batches = num_batches - num_val_batches
val_ds = ds.take(num_val_batches).repeat()
train_ds = ds.skip(num_val_batches).repeat()

# --- START BUILDING THE MODEL
import time
from keras import layers

@keras.saving.register_keras_serializable()
class TransformerDecoder(keras.Layer):
    def __init__(self, hidden_dim, intermediate_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        key_dim = hidden_dim // num_heads
        # Self-attention layers
        self.self_attention = layers.MultiHeadAttention(
            num_heads, key_dim, dropout=0.1
        )
        self.self_attention_layernorm = layers.LayerNormalization()
        # Feedforward layers
        self.feed_forward_1 = layers.Dense(intermediate_dim, activation="relu")
        self.feed_forward_2 = layers.Dense(hidden_dim)
        self.feed_forward_layernorm = layers.LayerNormalization()
        self.dropout = layers.Dropout(0.1)

    def call(self, inputs):
        # Self-attention computation
        residual = x = inputs
        x = self.self_attention(query=x, key=x, value=x, use_causal_mask=True)
        x = self.dropout(x)
        x = x + residual
        x = self.self_attention_layernorm(x)
        # Feedforward computation
        residual = x
        x = self.feed_forward_1(x)
        x = self.feed_forward_2(x)
        x = self.dropout(x)
        x = x + residual
        x = self.feed_forward_layernorm(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
            }
        )
        return config

from keras import ops

@keras.saving.register_keras_serializable()
class PositionalEmbedding(keras.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.token_embeddings = layers.Embedding(input_dim, output_dim)
        self.position_embeddings = layers.Embedding(sequence_length, output_dim)

    def call(self, inputs, reverse=False):
        if reverse:
            token_embeddings = self.token_embeddings.embeddings
            return ops.matmul(inputs, ops.transpose(token_embeddings))
        positions = ops.cumsum(ops.ones_like(inputs), axis=-1) - 1
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
            }
        )
        return config


# --- START STACKING EVERYTING UP
# Enables mixed precision (see chapter 18)
keras.config.set_dtype_policy("mixed_float16")

vocab_size = tokenizer.vocabulary_size()
hidden_dim = 512
intermediate_dim = 2056
num_heads = 8
num_layers = 8

inputs = keras.Input(shape=(None,), dtype="int32", name="inputs")
embedding = PositionalEmbedding(sequence_length, vocab_size, hidden_dim)
x = embedding(inputs)
x = layers.LayerNormalization()(x)
for i in range(num_layers):
    x = TransformerDecoder(hidden_dim, intermediate_dim, num_heads)(x)
outputs = embedding(x, reverse=True)
mini_gpt = keras.Model(inputs, outputs)

# --- LEARNING RATE WARMUP SCHEDULE
# Why do we need Learning Rate Warmup for deep Transformers?
#
# 1. Exploding gradients at the start of training:
#    At step 0, all weights in the model are completely random:
#    - PositionalEmbedding outputs random vectors,
#    - 8 stacked TransformerDecoder blocks multiply those vectors by random Attention matrices,
#    - At the end, the model guesses random words out of ~30,000 vocabulary tokens,
#      producing a huge initial loss.
#    When this huge error backpropagates through all stacked Transformer layers,
#    the initial gradients are very large and chaotic. Applying the full learning rate
#    (e.g. 1e-4) right away causes massive weight updates, making the loss diverge
#    and gradients explode to NaN (exploding gradients).
#
# 2. Solution — Linear Warmup:
#    Instead of starting with the full learning rate at step 0, we linearly ramp it up
#    from 0.0 to the target learning rate over the first `warmup_steps` batches (e.g. 2000 steps):
#
#      learning_rate
#      0.0001 |          /---------------------- (full learning rate)
#             |         /
#             |        /   <- warmup: linear growth
#             |       /       over the first ~2000 batches
#           0 +------/--------------------------> batch number (step)
#             0     2000
#
#    During the first ~2000 steps, weights are updated very gently, giving
#    LayerNormalization and Attention layers time to stabilize before full-speed training begins.
@keras.saving.register_keras_serializable()
class WarmupSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, rate=2e-4, warmup_steps=1_000.0):
        # Peak learning rate
        self.rate = rate
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = ops.cast(step, dtype="float32")
        scale = ops.minimum(step / self.warmup_steps, 1.0)
        return self.rate * scale

    def get_config(self):
        return {
            "rate": self.rate,
            "warmup_steps": self.warmup_steps,
        }

schedule = WarmupSchedule()

# --- START TRAINING
num_epochs = 8
# Set these to a lower value if you don't want to wait for training.
steps_per_epoch = num_train_batches // num_epochs
validation_steps = num_val_batches

# Ensure Keras knows the exact step count for progress bar & ETA calculation
# (same approach as in dots_ai.nosync/ml/training_pipeline.py):
#   12/7280 ━━━━━━━━━━━━━━━━━━━━ X:XX:XX Xs/step
print(
    f"Dataset split: {num_train_batches:,} training batches "
    f"({steps_per_epoch:,} steps/epoch x {num_epochs} epochs), "
    f"{validation_steps:,} validation batches"
)


def _format_duration(seconds):
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}h {minutes:02d}m {secs:02d}s"


class TrainingTimeEstimator(keras.callbacks.Callback):
    """Estimates and prints 1-epoch and total training duration."""

    def __init__(self, steps_per_epoch, total_epochs, sample_steps=10):
        super().__init__()
        self.steps_per_epoch = steps_per_epoch
        self.total_epochs = total_epochs
        self.sample_steps = sample_steps
        self.epoch_start_time = None
        self.steady_start_time = None
        self.estimate_printed = False

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.perf_counter()

    def on_train_batch_end(self, batch, logs=None):
        # Skip batch 0 (graph compilation) when estimating steady-state step time
        if batch == 1 and not self.estimate_printed:
            self.steady_start_time = time.perf_counter()
        elif (
            batch == self.sample_steps + 1
            and not self.estimate_printed
            and self.steady_start_time is not None
        ):
            elapsed = time.perf_counter() - self.steady_start_time
            sec_per_step = elapsed / self.sample_steps
            est_epoch_sec = sec_per_step * self.steps_per_epoch
            est_total_sec = est_epoch_sec * self.total_epochs
            print(
                f"\n[Time Estimate] ~{sec_per_step:.2f}s/step | "
                f"1 epoch ({self.steps_per_epoch:,} steps): ~{_format_duration(est_epoch_sec)} | "
                f"Full training ({self.total_epochs} epochs): ~{_format_duration(est_total_sec)}"
            )
            self.estimate_printed = True

    def on_epoch_end(self, epoch, logs=None):
        epoch_sec = time.perf_counter() - self.epoch_start_time
        remaining_epochs = self.total_epochs - (epoch + 1)
        remaining_sec = epoch_sec * remaining_epochs
        print(
            f"\n[Epoch {epoch + 1}/{self.total_epochs}] "
            f"Duration: {_format_duration(epoch_sec)} | "
            f"Estimated remaining time: {_format_duration(remaining_sec)}"
        )


MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "mini_gpt"
    / "mini_gpt.keras"
)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

HISTORY_PLOT_PATH = (
    PROJECT_DIR
    / "docs"
    / "imgs"
    / "mini_gpt"
    / "training_history.png"
)
HISTORY_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        save_best_only=True,
        monitor="val_loss",
    ),
    TrainingTimeEstimator(
        steps_per_epoch=steps_per_epoch,
        total_epochs=num_epochs,
    ),
]

mini_gpt.compile(
    optimizer=keras.optimizers.Adam(schedule),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
history = mini_gpt.fit(
    train_ds,
    validation_data=val_ds,
    epochs=num_epochs,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=callbacks,
)

# --- START MODEL VERIFICATION (PLOTTING ACCURACY & LOSS)
import matplotlib.pyplot as plt

accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)

figure, (acc_ax, loss_ax) = plt.subplots(1, 2, figsize=(12, 5))

acc_ax.plot(epochs, accuracy, "r--", label="Training accuracy")
acc_ax.plot(epochs, val_accuracy, "b", label="Validation accuracy")
acc_ax.set_title("Training and validation accuracy")
acc_ax.set_xlabel("Epoch")
acc_ax.set_ylabel("Accuracy")
acc_ax.legend()

loss_ax.plot(epochs, loss, "r--", label="Training loss")
loss_ax.plot(epochs, val_loss, "b", label="Validation loss")
loss_ax.set_title("Training and validation loss")
loss_ax.set_xlabel("Epoch")
loss_ax.set_ylabel("Loss")
loss_ax.legend()

figure.tight_layout()
figure.savefig(HISTORY_PLOT_PATH, dpi=300)
plt.show()
# --- END MODEL VERIFICATION

if __name__ == "__main__":
    # Turn on debugging
    # first_batch = next(iter(ds))
    pass