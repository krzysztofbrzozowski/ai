from keras import ops

metric = keras.metrics.SparseCategoricalAccuracy()
targets = ops.array([0, 1, 2])
predictions = ops.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# Manual call of the metrics function from Keras API
metric.update_state(targets, predictions)
current_result = metric.result()
print(f"result: {current_result:.2f}")
