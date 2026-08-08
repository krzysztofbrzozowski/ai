# Table of Contents
- [Tensors and vectors](#tensors-and-vectors)
- [Basic network components](#basic-network-components)
- [01_minst_recognition_network.py graph](#01_minst_recognition_network-graph)

# Tensors and vectors
## 0D tensor (scalar), rank-0 tensor
>>> x = np.array(12)
>>> x.ndim
0

## 5D vector but it is 1D tensor, rank-1 tensor
>>> x = np.array([12, 3, 6, 14, 7])
>>> x.ndim
1

## 2D tensor, rank-2 tensor 
>>> x = np.array([[5, 78, 2, 34, 0],
...               [6, 79, 3, 35, 1],
...               [7, 80, 4, 36, 2]])
>>> x.ndim
2

## rank-3 tensor
>>> x = np.array([[[5, 78, 2, 34, 0],
...                [6, 79, 3, 35, 1],
...                [7, 80, 4, 36, 2]],
...               [[5, 78, 2, 34, 0],
...                [6, 79, 3, 35, 1],
...                [7, 80, 4, 36, 2]],
...               [[5, 78, 2, 34, 0],
...                [6, 79, 3, 35, 1],
...                [7, 80, 4, 36, 2]]])
>>> x.ndim
3

## Examples
### An actuarial dataset of people, where we consider each person’s age, gender, and income. Each person can be characterized as a vector of three values, and thus an entire dataset of 100,000 people can be stored in a rank-2 tensor of shape


people = np.array([
    [25, 0, 50000],  # 25 lat, mężczyzna, 50k
    [32, 1, 72000],  # 32 lata, kobieta, 72k
    [41, 0, 90000],  # 41 lat, mężczyzna, 90k
    ...
    [22, 1, 90000],
])

people.shape
-> (100000, 3)

---
### A dataset of stock prices — Every minute, we store the current price of the stock, the highest price in the past minute, and the lowest price in the past minute. Thus every minute is encoded as a 3D vector, an entire day of trading is encoded as a matrix of shape (390, 3) (there are 390 minutes in a trading day), and 250 days’ worth of data can be stored in a rank-3 tensor of shape (250, 390, 3). Here, each sample would be one day’s worth of data

Dzień 1
[
  [current, high, low],
  [current, high, low],
  ...
]

Dzień 2
[
  [current, high, low],
  [current, high, low],
  ...
]

...

Dzień 250s
[
  [current, high, low],
  [current, high, low],
  ...
]


250 dni
 └─ każdy dzień ma 390 minut
      └─ każda minuta ma 3 liczb

-> stock_data[10, 50, 1]
10 → 11. dzień
50 → 51. minuta tego dnia
1 → druga cecha (high)


### A dataset of tweets, where we encode each tweet as a sequence of 280 characters out of an alphabet of 128 unique characters — In this setting, each character can be encoded as a binary vector of size 128 (an all-zeros vector except for a 1 entry at the index corresponding to the character). Then each tweet can be encoded as a rank-2 tensor of shape (280, 128), and a dataset of 1 million tweets can be stored in a tensor of shape (1000000, 280, 128)
> Used 4 digits instead of 128 to visualize
[
    Tweet 0
    [
    [0,1,0,0],  # b
    [1,0,0,0],  # a
    [0,0,0,1]   # d
    ]

    Tweet 1
    [
    [0,0,1,0],  # c
    [1,0,0,0],  # a
    [0,1,0,0]   # b
    ]
    ...
    Tweet 1000000s
    [
    [0,0,1,0],  # c
    [1,0,0,0],  # a
    [0,1,0,0]   # b
    ]
]

tweets = np.array([
    [
        [0,1,0,0],
        [1,0,0,0],
        [0,0,0,1]
    ],
    [
        [0,0,1,0],
        [1,0,0,0],
        [0,1,0,0]
    ]
    ...
    [
        [0,0,1,0],
        [1,0,0,0],
        [0,1,0,0]
    ]
])

# Basic network components
```python
keras.layers.Dense(512, activation="relu")
```
```bash
-> output = relu(matmul(input, W) + b)
```
Let’s unpack this. We have three tensor operations here:

- A tensor product (**matmul**) between the input tensor and a tensor named W.
- An addition (+) between the resulting matrix and a vector b.
- A relu operation: **relu(x)** is **max(x, 0)**. "relu" stands for “REctified Linear Unit.”

# Gradient function
grad(y, x)
->
x=[x1 ​x2​​], y=x1**2 ​+ 3x2​
grad(y,x) = ∇x​y = [​∂y/∂x1 ​∂y/∂x2​​​]=[2x1 ​3​]

### Simple example
```python
def fg(x):
    x1 = g(x)
    y = f(x1)
    return y

grad(y, x) == grad(y, x1) * grad(x1, x)
```

### Simple example 2
```python
def fghj(x):
    x1 = j(x)
    x2 = h(x1)
    x3 = g(x2)
    y = f(x3)
    return y

grad(y, x) == grad(y, x3) * grad(x3, x2) * grad(x2, x1) * grad(x1, x)
```

# 01_minst_recognition_network graph

![A first computation graph](https://deeplearningwithpython.io/images/ch02/a_first_computation_graph.90dec1fc.png)

*Source: François Chollet and Matthew Watson,  
[Deep Learning with Python — Chapter 2](https://deeplearningwithpython.io/chapters/chapter02_mathematical-building-blocks/).*


## Backpropagation
![Path from loss_val to w in the backward graph](https://deeplearningwithpython.io/images/ch02/path_in_backward_graph.fe91e7d0.png)

*Source: François Chollet and Matthew Watson, Deep Learning with Python, Chapter 2.*
For instance
```python
grad(loss_val, w) = grad(loss_val, x2) * grad(x2, x1) * grad(x1, w)
```
->
Backpropagation (grad) values from the example
```python
grad(loss_val, w) = 1 * 1 * 2 = 2
grad(loss_val, b) = 1 * 1 = 1
```

### Deep learning in three figures

<p align="center">
  <img
    src="https://deeplearningwithpython.io/images/ch02/deep-learning-in-3-figures-3_alt.40aa865d.png"
    alt="Deep learning in three figures"
    width="800"
  >
</p>

<p align="center">
  <em>
    Source: François Chollet and Matthew Watson,
    Deep Learning with Python, Chapter 2.
  </em>
</p>

## Functions relation chain
### Create model
> Below you can find some sequence
> Look at the comments in 01_manual_implementation_recognition_nerwork.py for more details
```python
    model = NaiveSequential(
        [
            NaiveDense(input_size=28 * 28, output_size=512, activation=ops.relu),
            NaiveDense(input_size=512, output_size=10, activation=ops.softmax),
        ]
    )
```
-> see implementation of NaiveSequential and NaiveDense
↓
```python
def fit(model, images, labels, epochs, batch_size=128):
  ...
  for batch_counter in range(batch_generator.num_batches):
    loss = one_training_step(model, images_batch, labels_batch)
↓
```python
one_training_step(model, images_batch, labels_batch):
    with tf.GradientTape() as tape:
        predictions = model(images_batch)
        loss = ops.sparse_categorical_crossentropy(labels_batch, predictions)
        average_loss = ops.mean(loss)
    # 'tape' gradient calculation for backpropagation
    gradients = tape.gradient(average_loss, model.weights)
    update_weights(gradients, model.weights)
```
↓
```python
optimizer = optimizers.SGD(learning_rate=1e-3)

def update_weights(gradients, weights):
    optimizer.apply_gradients(zip(gradients, weights))
```