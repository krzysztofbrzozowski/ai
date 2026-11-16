import os
import sys

# Silence debugpy/pydevd "did not finish after 3.00 seconds" warning in Debug Console
for mod_name, mod in list(sys.modules.items()):
    if "pydevd" in mod_name and hasattr(mod, "PYDEVD_WARN_EVALUATION_TIMEOUT"):
        setattr(mod, "PYDEVD_WARN_EVALUATION_TIMEOUT", 3600.0)

# Why we use the PyTorch backend ("torch") instead of TensorFlow ("tensorflow") here:
# 1. No XLA compiler support in `tensorflow-metal`:
#    During autoregressive text generation (`generate()`), Transformer attention layers
#    update the KV-cache at each token step via `keras.ops.slice_update`.
#    In the TensorFlow backend, `slice_update` calls `XlaDynamicUpdateSlice` (an XLA op),
#    and `CausalLM.generate()` wraps `generate_step` in `tf.function(jit_compile=True)`.
#    Because Apple's `tensorflow-metal` PluggableDevice does not register an XLA compiler
#    platform for METAL, execution fails with:
#    `NOT_FOUND: could not find registered platform with id: ... [Op:XlaDynamicUpdateSlice]`.
# 2. Native mutable tensor indexing on Apple Metal (`mps`) in PyTorch:
#    In the PyTorch backend, `keras.ops.slice_update` uses native in-place slice assignment
#    (`outputs[slices] = updates` / `index_put_`) and runs under `torch.no_grad()` directly
#    on Apple Silicon Metal Performance Shaders (`mps`) without requiring XLA.
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers, ops
import keras_hub
import numpy as np
import tensorflow as tf

gemma_lm = keras_hub.models.Gemma4CausalLM.from_preset(
    "gemma4_2b",
    dtype="float32",
)

# gemma_lm.summary()
# Preprocessor: "gemma4_causal_lm_preprocessor"
#  Total params: 5,104,298,467 (19.01 GB)
#  Trainable params: 5,104,186,912 (19.01 GB)
#  Non-trainable params: 111,555 (435.76 KB)

# >>> gemma_lm.compile(sampler="greedy")
# >>> print(gemma_lm.generate("A piece of advice", max_length=40))
# >>> gemma_lm.generate("How can I make brownies?", max_length=40)

# --- START DATA FOR FINE TUNING
import json

PROMPT_TEMPLATE = """"[instruction]\n{}[end]\n[response]\n"""
RESPONSE_TEMPLATE = """{}[end]"""

dataset_path = keras.utils.get_file(
    origin=(
        "https://hf.co/datasets/databricks/databricks-dolly-15k/"
        "resolve/main/databricks-dolly-15k.jsonl"
    ),
)
data = {"prompts": [], "responses": []}
with open(dataset_path) as file:
    for line in file:
        features = json.loads(line)
        if features["context"]:
            continue
        data["prompts"].append(PROMPT_TEMPLATE.format(features["instruction"]))
        data["responses"].append(RESPONSE_TEMPLATE.format(features["response"]))

# --- START OF DATASET
ds = tf.data.Dataset.from_tensor_slices(data).shuffle(2000).batch(2)
val_ds = ds.take(100)
train_ds = ds.skip(100)

# --- START ENABLE LORA
gemma_lm.backbone.enable_lora(rank=8)

# --- COMPILE AND TRAIN
gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(5e-5),
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
gemma_lm.fit(train_ds, validation_data=val_ds, epochs=1)

if __name__ == "__main__":
    pass
