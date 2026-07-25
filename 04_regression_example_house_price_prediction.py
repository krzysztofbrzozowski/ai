# --- START DATASET
from keras.datasets import california_housing

# Make sure to pass version="small" to get the right dataset.
(train_data, train_targets), (test_data, test_targets) = (
    california_housing.load_data(version="small")
)
# >>> train_data.shape
# (480, 8)
# >>> test_data.shape
# (120, 8)

# Note:
# Normalize the data
# Compute the mean and standard deviation only on the training data
# Then use these same values to normalize both the training set and the test set
# We avoid using any statistics from the test data, so the test set remains truly unseen
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
x_train = (train_data - mean) / std
x_test = (test_data - mean) / std

# Note:
# Scale the target values to a smaller range
# The input features are normalized around 0, and the model starts with small random weights
# Its initial predictions are therefore also small
# If targets remain in the range of $60,000 to $500,000, the model would need very large weights
# This could make training unnecessarily slow
# Dividing targets by 100,000 changes the range to approximately 0.6–5.0
# Multiply predictions by 100,000 later to convert them back to dollar values
y_train = train_targets / 100000
y_test = test_targets / 100000
# --- END DATASET
# --- START MODEL
import keras
from keras import layers

def get_model():
    # Note:
    # Use a small model because the training dataset contains relatively few samples
    # With limited data, a large model can easily memorize the training examples and overfit
    # Two hidden layers with 64 units provide enough capacity while reducing the risk of overfitting
    model = keras.Sequential(
        [
            layers.Dense(64, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(1),
        ]
    )
    # Use a single output unit with no activation for scalar regression
    # A linear output allows the model to predict any continuous value
    # An activation such as sigmoid would restrict predictions to the range 0 to 1
    # Use mean squared error as the loss function for the regression task
    # MSE measures the squared difference between predictions and target values
    # -> Monitor mean absolute error to measure the average prediction error
    # -> Since targets were divided by 100,000, an MAE of 0.5 means an average error of $50,000
    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mean_absolute_error"],
    )
    return model
# -- END MODEL

# --- START TRAINING USING K-FOLD VALIDATION
import numpy as np

k = 4
num_val_samples = len(x_train) // k
num_epochs = 50
all_scores = []

for i in range(k):
    print(f"Processing fold #{i + 1}")
    # Prepares the validation data: data from partition #k
    fold_x_val = x_train[i * num_val_samples : (i + 1) * num_val_samples]
    fold_y_val = y_train[i * num_val_samples : (i + 1) * num_val_samples]

    # np.concatenate:
    # Combine multiple NumPy arrays into a single array
    # axis=0 adds more rows
    # Example: [[1, 2], [3, 4]] + [[5, 6]] -> [[1, 2], [3, 4], [5, 6]]
    # axis=1 adds more columns
    # Example: [[1, 2], [3, 4]] + [[5], [6]] -> [[1, 2, 5], [3, 4, 6]]
    #
    # ---
    #
    # Example for 480 samples split into 4 folds
    #
    # num_val_samples = 120
    #
    # i = 0
    # validation range: x_train[0:120] -> fold_x_val
    # training ranges: x_train[:0] and x_train[120:] -> fold_x_train
    #
    # i = 1
    # validation range: x_train[120:240] -> fold_x_val
    # training ranges: x_train[:120] and x_train[240:] -> fold_x_train
    #
    # i = 2
    # validation range: x_train[240:360] -> fold_x_val
    # training ranges: x_train[:240] and x_train[360:] -> fold_x_train
    #
    # i = 3
    # validation range: x_train[360:480] -> fold_x_val
    # training ranges: x_train[:360] and x_train[480:] -> fold_x_train
    # -> see https://deeplearningwithpython.io/images/ch04/3-fold-cross-validation.40bb5356.png
    #
    # np.concatenate joins the two training ranges into one array
    # The current validation range is excluded from training
    fold_x_train = np.concatenate(
        [x_train[: i * num_val_samples], x_train[(i + 1) * num_val_samples :]],
        axis=0,
    )
    fold_y_train = np.concatenate(
        [y_train[: i * num_val_samples], y_train[(i + 1) * num_val_samples :]],
        axis=0,
    )

    # Builds the Keras model (already compiled)
    model = get_model()
    # Trains the model
    model.fit(
        fold_x_train,
        fold_y_train,
        epochs=num_epochs,
        batch_size=16,
        verbose=0,
    )
    # Evaluates the model on the validation data
    scores = model.evaluate(fold_x_val, fold_y_val, verbose=0)
    val_loss, val_mae = scores
    all_scores.append(val_mae)
# --- END TRAINING USING K-FOLD VALIDATION
# --- START EVALUATE MODEL ON TEST DATA
# Results from training for 50 epochs on the entire training data
"""
-> Results from K-fold validation:
>>> [round(value, 3) for value in all_scores]
[0.305, 0.291, 0.25, 0.316]
>>> round(np.mean(all_scores), 3)
0.289
"""
# Note:
# Validation MAE varies across folds from about 0.232 to 0.349
# A single fold score may therefore be misleading
# The average MAE across all folds is more reliable because it reduces the effect of one particular data split
# The average MAE is about 0.296
# Since targets were scaled by 100,000, this corresponds to an average prediction error of about $29,600
# This error is significant because house prices range from about $60,000 to $500,000

### --- START TRAIN AND EVALUATE MODEL ON TEST DATA (EPOCHS = 200)
k = 4
num_val_samples = len(x_train) // k
num_epochs = 200
all_mae_histories = []
for i in range(k):
    print(f"Processing fold #{i + 1}")
    # Prepares the validation data: data from partition #k
    fold_x_val = x_train[i * num_val_samples : (i + 1) * num_val_samples]
    fold_y_val = y_train[i * num_val_samples : (i + 1) * num_val_samples]
    # Prepares the training data: data from all other partitions
    fold_x_train = np.concatenate(
        [x_train[: i * num_val_samples], x_train[(i + 1) * num_val_samples :]],
        axis=0,
    )
    fold_y_train = np.concatenate(
        [y_train[: i * num_val_samples], y_train[(i + 1) * num_val_samples :]],
        axis=0,
    )
    # Builds the Keras model (already compiled)
    model = get_model()
    # Trains the model
    history = model.fit(
        fold_x_train,
        fold_y_train,
        validation_data=(fold_x_val, fold_y_val),
        epochs=num_epochs,
        batch_size=16,
        verbose=0,
    )
    mae_history = history.history["val_mean_absolute_error"]
    all_mae_histories.append(mae_history)

    # Calculate the average of the per-epoch MAE scores for all folds
    average_mae_history = [
        np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)
    ]

    # Plot the per-epoch MAE scores for all folds
    import matplotlib.pyplot as plt
    epochs = range(1, len(average_mae_history) + 1)
    plt.plot(epochs, average_mae_history)
    plt.xlabel("Epochs")
    plt.ylabel("Validation MAE")
    plt.show()
    # -> see the ~results https://deeplearningwithpython.io/images/ch04/california_housing_validation_mae_plot.af306c57.png
    truncated_mae_history = average_mae_history[10:]
    epochs = range(10, len(truncated_mae_history) + 10)
    plt.plot(epochs, truncated_mae_history)
    plt.xlabel("Epochs")
    plt.ylabel("Validation MAE")
    plt.show()
    # -> see the ~results https://deeplearningwithpython.io/images/ch04/california_housing_validation_mae_plot_zoomed.928f390d.png
### --- END TRAIN AND EVALUATE MODEL ON TEST DATA (EPOCHS = 200)
### --- START FINAL TRAINING AND EVALUATION ON TEST DATA
# Gets a fresh, compiled model
model = get_model()
# Trains it on the entirety of the data
model.fit(x_train, y_train, epochs=130, batch_size=16, verbose=0)
test_mean_squared_error, test_mean_absolute_error = model.evaluate(
    x_test, y_test
)
"""
>>> round(test_mean_absolute_error, 3)
0.31
"""
### --- END FINAL TRAINING AND EVALUATION ON TEST DATA
### --- START PREDICTIONS ON TEST DATA
"""
>>> predictions = model.predict(x_test)
>>> predictions[0]
array([2.834494], dtype=float32)
"""
### --- END PREDICTIONS ON TEST DATA


if __name__ == "__main__":
    pass