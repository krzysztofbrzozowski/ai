import pathlib
import keras
from keras import layers, ops
import keras_hub
import numpy as np

# Note: mini_gpt/main_test.py is 1 directory below AI.nosync root, so we use two .parent calls
PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent

# Enable mixed_float16 policy (same policy that the model was trained and saved with)
keras.config.set_dtype_policy("mixed_float16")


# Custom classes must be registered before calling keras.models.load_model()
@keras.saving.register_keras_serializable()
class TransformerDecoder(keras.Layer):
    def __init__(self, hidden_dim, intermediate_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        key_dim = hidden_dim // num_heads
        self.self_attention = layers.MultiHeadAttention(
            num_heads, key_dim, dropout=0.1, dtype=self.dtype_policy
        )
        self.self_attention_layernorm = layers.LayerNormalization(
            dtype=self.dtype_policy
        )
        self.feed_forward_1 = layers.Dense(
            intermediate_dim, activation="relu", dtype=self.dtype_policy
        )
        self.feed_forward_2 = layers.Dense(hidden_dim, dtype=self.dtype_policy)
        self.feed_forward_layernorm = layers.LayerNormalization(
            dtype=self.dtype_policy
        )
        self.dropout = layers.Dropout(0.1, dtype=self.dtype_policy)

    def call(self, inputs):
        residual = x = inputs
        x = self.self_attention(query=x, key=x, value=x, use_causal_mask=True)
        x = self.dropout(x)
        x = x + residual
        x = self.self_attention_layernorm(x)
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


@keras.saving.register_keras_serializable()
class PositionalEmbedding(keras.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.token_embeddings = layers.Embedding(
            input_dim, output_dim, dtype=self.dtype_policy
        )
        self.position_embeddings = layers.Embedding(
            sequence_length, output_dim, dtype=self.dtype_policy
        )

    def call(self, inputs, reverse=False):
        if reverse:
            token_embeddings = ops.cast(
                self.token_embeddings.embeddings, inputs.dtype
            )
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


@keras.saving.register_keras_serializable()
class WarmupSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, rate=2e-4, warmup_steps=1_000.0):
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


# --- START GET TOKENIZER
vocabulary_file = keras.utils.get_file(
    origin="https://hf.co/mattdangerw/spiece/resolve/main/vocabulary.proto",
    cache_dir=PROJECT_DIR,
)
tokenizer = keras_hub.tokenizers.SentencePieceTokenizer(vocabulary_file)

# --- START LOAD MODEL
MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "mini_gpt"
    / "mini_gpt.keras"
)
mini_gpt = keras.models.load_model(MODEL_PATH)


# --- FUNCTION TO GENERATE TEXT FROM PROMPT
def generate(prompt, max_length=64):
    tokens = list(ops.convert_to_numpy(tokenizer(prompt)))
    prompt_length = len(tokens)
    # Generate 1 token in each loop iteration until total length reaches max_length.
    # E.g. if prompt has 10 tokens, we iterate 54 times to reach 64 tokens in total.
    for _ in range(max_length - prompt_length):
        prediction = mini_gpt(ops.convert_to_numpy([tokens]))
        prediction = ops.convert_to_numpy(prediction[0, -1])
        tokens.append(np.argmax(prediction).item())
    return tokenizer.detokenize(tokens)


if __name__ == "__main__":
    pass

# >>> generate("Hello, I’m")