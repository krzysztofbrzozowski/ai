# TensorFlow with Metal GPU on Apple Silicon

## Requirements

- macOS 12.0+ on Apple Silicon (M1/M2/M3/M4)
- Python 3.12 (tensorflow-metal does not yet support Python 3.13)
- `tensorflow-macos` (Apple's fork with Metal PluggableDevice support)
- `tensorflow-metal` (Metal GPU plugin)

## Setup

### 1. Create virtual environment with Python 3.12

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install packages

```bash
pip install tensorflow==2.18.1 tensorflow-metal==1.2.0 tensorflow-text==2.18.1 keras==3.15.1 keras-hub==0.19.0
```

Or from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Upgrade `keras-hub` for newer models (Gemma 3 / Gemma 4)

Because `keras-hub` 0.19.1+ declares a dependency on `tensorflow-text>=2.20` (which would force a TensorFlow upgrade and break `tensorflow-metal`), upgrade `keras-hub` **without dependencies** after installing the base packages:

```bash
pip install --upgrade --no-deps keras-hub
```

This installs the latest `keras-hub` (e.g. `0.33.0` with `gemma3_*` and `gemma4_*` presets) while keeping `tensorflow==2.18.1`, `tensorflow-metal==1.2.0`, and `tensorflow-text==2.18.1` intact.

### 4. Verify GPU and `keras-hub` presets

```bash
python3 -c "import tensorflow as tf, keras_hub; print('GPUs:', tf.config.list_physical_devices('GPU')); print('KerasHub:', keras_hub.__version__)"
```

Expected output should include:

```
Metal device set to: Apple M4 Pro
GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
KerasHub: 0.33.0
```

## Notes

- `tensorflow==2.18.1` (or `tensorflow-macos==2.16.2`) is the latest version compatible with `tensorflow-metal==1.2.0`. Newer TensorFlow versions (2.19+) do not have a compatible `tensorflow-metal` wheel.
- Python 3.13 is not supported by `tensorflow-metal` (no `cp313` wheel available). The Metal plugin supports Python 3.9–3.12.
- During a normal `pip install`, `keras-hub` must be pinned to `<= 0.19.0` because `0.19.1+` requires `tensorflow-text>=2.20`. To use newer `keras-hub` releases (e.g. `0.33.0` for Gemma 3 / Gemma 4), always install or upgrade it with `--no-deps` (`pip install --upgrade --no-deps keras-hub`).
- **Why use `KERAS_BACKEND="torch"` for LLM text generation (`CausalLM.generate()`)**:
  During autoregressive generation, `keras-hub` updates the Transformer KV-cache at each step via `keras.ops.slice_update`. In the TensorFlow backend, `slice_update` invokes `XlaDynamicUpdateSlice` (an XLA op), and `CausalLM.generate()` wraps `generate_step` in `tf.function(jit_compile=True)`. Because `tensorflow-metal` does not register an XLA compiler platform for `METAL`, `generate()` fails with `NOT_FOUND: could not find registered platform ... [Op:XlaDynamicUpdateSlice]`. Setting `os.environ["KERAS_BACKEND"] = "torch"` before importing `keras` runs `slice_update` as native PyTorch tensor slice assignment on Apple Metal (`mps`) under `torch.no_grad()` without requiring XLA.
