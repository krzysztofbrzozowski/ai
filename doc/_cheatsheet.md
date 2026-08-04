# Cheatsheet

## Choosing the output layer, loss function, and metrics

| Task | Last-layer activation | Loss function | Typical metrics |
|---|---|---|---|
| Binary classification | `sigmoid` | `binary_crossentropy` | `binary_accuracy`, `ROC AUC` |
| Multiclass, single-label classification | `softmax` | `categorical_crossentropy` or `sparse_categorical_crossentropy` | `categorical_accuracy`, `top-k categorical accuracy`, `ROC AUC` |
| Multiclass, multi-label classification | `sigmoid` | `binary_crossentropy` | `binary_accuracy`, `ROC AUC` |
| Regression | None / linear activation | `mean_squared_error` | `mean_absolute_error` |

## Keras examples

### Binary classification

```python
outputs = layers.Dense(1, activation="sigmoid")(x)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["binary_accuracy"],
)
```

### Multiclass, single-label classification

Use `sparse_categorical_crossentropy` when labels are integers such as `0`, `1`, `2`, ..., `9`.

```python
outputs = layers.Dense(10, activation="softmax")(x)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
```

Use `categorical_crossentropy` when labels are one-hot encoded.

```python
outputs = layers.Dense(10, activation="softmax")(x)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy"],
)
```

### Multiclass, multi-label classification

Each class is predicted independently, so the output uses one sigmoid per class.

```python
outputs = layers.Dense(num_classes, activation="sigmoid")(x)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["binary_accuracy"],
)
```

### Regression

The final layer usually has no activation function.

```python
outputs = layers.Dense(1)(x)

model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mean_absolute_error"],
)
```

## Quick rule

```text
Two classes
→ Dense(1, sigmoid)
→ binary_crossentropy

Many mutually exclusive classes
→ Dense(number_of_classes, softmax)
→ sparse_categorical_crossentropy or categorical_crossentropy

Many independent labels
→ Dense(number_of_labels, sigmoid)
→ binary_crossentropy

Continuous numerical value
→ Dense(1)
→ mean_squared_error
```
