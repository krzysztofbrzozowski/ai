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
pip install tensorflow-macos==2.16.2 tensorflow-metal==1.2.0 keras==3.15.1 keras-hub==0.16.1
```

Or from requirements.txt:

```bash
pip install -r requirements.txt
```

### 3. Verify GPU is detected

```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Expected output should include:

```
Metal device set to: Apple M4 Pro
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## Notes

- `tensorflow-macos` 2.16.2 is currently the latest version compatible with `tensorflow-metal`. Newer TensorFlow versions (2.17+) do not yet have built-in Metal support for macOS.
- Python 3.13 is not supported by `tensorflow-metal` (no cp313 wheel available).
- If you need a different Python version, the metal plugin supports Python 3.9–3.12.
- `keras-hub` must be pinned to 0.16.1 — newer versions (0.17+) depend on `tensorflow-text>=2.20` which forces a TensorFlow upgrade and breaks Metal GPU support.
