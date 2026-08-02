import keras
from keras import layers

# --- START MODEL DEFINITION
# System to rank customer support tickets by priority and route them to the appropriate department
# INPUTS (3):
# -> The title of the ticket (text input)
# -> The text body of the ticket (text input)
# -> Any tags added by the user (categorical input, assumed here to be multi-hot encoded)
# OUTPUTS (2):
# -> The priority score of the ticket, a scalar between 0 and 1 (sigmoid output)
# -> The department that should handle the ticket (a softmax over the set of departments)
# Note:
# in 03_from_layers_to_models_using_keras.py we instantieded keras.Layer
# and now we create layers from base and instantiete keras.Model
# It is similar to Sequential but we defining our own flow and data structure
class CustomerTicketModel(keras.Model):
    def __init__(self, num_departments):
        # Don't forget to call the super constructor!
        super().__init__()
        # Defines sublayers in the constructor
        self.concat_layer = layers.Concatenate()
        self.mixing_layer = layers.Dense(64, activation="relu")
        self.priority_scorer = layers.Dense(1, activation="sigmoid")
        self.department_classifier = layers.Dense(
            num_departments, activation="softmax"
        )

    # Defines the forward pass in the call() method
    def call(self, inputs):
        title = inputs["title"]
        text_body = inputs["text_body"]
        tags = inputs["tags"]

        features = self.concat_layer([title, text_body, tags])
        features = self.mixing_layer(features)
        priority = self.priority_scorer(features)
        department = self.department_classifier(features)
        # Returns the model's two final predictions
        # Keras compares these outputs with the true targets during training
        # This allows it to calculate the losses and update the model's weights
        # The same outputs are also returned by evaluate(), predict(), and model(inputs)
        return priority, department

# --- END MODEL DEFINITION
# --- START MODEL TRAINING
model = CustomerTicketModel(num_departments=4)

priority, department = model(
    {
        "title": title_data,
        "text_body": text_body_data,
        "tags": tags_data
    }
)

model.compile(
    optimizer="adam",
    loss={
        "priority": "mean_squared_error",
        "department": "sparse_categorical_crossentropy",
    },
    metrics={
        "priority": ["mean_absolute_error"],
        "department": ["accuracy"],
    },
)
model.fit(
    {
        "title": title_data,
        "text_body": text_body_data,
        "tags": tags_data
    },
    {
        "priority": priority_data,
        "department": department_data
    },
    epochs=1,
)
model.evaluate(
    {
        "title": title_data,
        "text_body": text_body_data,
        "tags": tags_data
    },
    {
        "priority": priority_data,
        "department": department_data
    },
    # To get dict as return this needs to be set
    return_dict=True,
)
priority_preds, department_preds = model.predict(
    {
        "title": title_data,
        "text_body": text_body_data,
        "tags": tags_data
    }
)
