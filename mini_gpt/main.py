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
# Verify the amount of batches in ds
# c = 0               # <- pierwszy argument reduce: stan początkowy (initial_state = 0)
# for _ in ds:        # <- drugi argument: lambda c, _: c + 1
#     c = c + 1
# return c            # -> zwraca tf.Tensor z końcową liczbą elementów

# W przypadku potoku z TextLineDataset -> interleave -> rebatch, TensorFlow nie wie z góry, ile linii tekstu i ile tokenów wyjdzie z 50 plików .txt na dysku:

# len(ds) rzuci błąd TypeError: The dataset length is unknown,
# ds.cardinality().numpy() zwróci -2 (UNKNOWN_CARDINALITY).

# Dlatego ds.reduce(0, lambda c, _: c + 1) to sposób na wymuszenie przejścia przez cały potok danych od początku do końca i policzenie, ile dokładnie batchy z niego powstanie.
# ---
# Kiedy wywołasz ds.reduce(0, lambda c, _: c + 1), zmuszasz komputer, żeby w tym momencie:
# Wczytał z dysku wszystkie 3,8 GB tekstu,
# Przemielił przez tokenizer na procesorze prawie miliard tokenów,
# Pociął je na okna po 256 tokenów i ułożył w 58 746 batchy,
# Po czym każdy batch wyrzucił do kosza, zwiększając tylko licznik c + 1.

# --- START TRAINING AND VALIDATION DATASET
num_batches = 58746
num_val_batches = 500
num_train_batches = num_batches - num_val_batches
val_ds = ds.take(num_val_batches).repeat()
train_ds = ds.skip(num_val_batches).repeat()

# --- START BUILDING THE MODEL
from keras import layers

class TransformerDecoder(keras.Layer):
    def __init__(self, hidden_dim, intermediate_dim, num_heads):
        super().__init__()
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

from keras import ops

class PositionalEmbedding(keras.Layer):
    def __init__(self, sequence_length, input_dim, output_dim):
        super().__init__()
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
# class WarmupSchedule(keras.optimizers.schedules.LearningRateSchedule):
class WarmupSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self):
        # Peak learning rate
        self.rate = 2e-4
        self.warmup_steps = 1_000.0

    def __call__(self, step):
        step = ops.cast(step, dtype="float32")
        scale = ops.minimum(step / self.warmup_steps, 1.0)
        return self.rate * scale

schedule = WarmupSchedule()
# --- START TRAINING
num_epochs = 8
# Set these to a lower value if you don't want to wait for training.
steps_per_epoch = num_train_batches // num_epochs
validation_steps = num_val_batches

mini_gpt.compile(
    optimizer=keras.optimizers.Adam(schedule),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
mini_gpt.fit(
    train_ds,
    validation_data=val_ds,
    epochs=num_epochs,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
)

if __name__ == "__main__":
    # Turn on debugging
    # first_batch = next(iter(ds))
    pass