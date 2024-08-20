
---

# Implementation of LeNet-5

We will use the [tensorflow.keras Sequential API](https://www.tensorflow.org/guide/keras/sequential_model) to build LeNet-5 from the original paper: “[Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)” by Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner.

[Video tutorial](https://www.youtube.com/watch?v=rFpzCPcI6O0&list=PLaPdEEY26UXyE3UchW0C742xh542yh0yI&index=1)

---

In the paper, we can read:

>**[i]** "The input image is 32x32 pixels, and is passed through a series of convolutional, pooling, and fully-connected layers."
>
>**[ii]** "The first convolutional layer uses 6 kernels of size 5x5 to produce 6 feature maps of size 28x28."
>
>**[iii]** "The pooling layers use a 2x2 kernel to reduce the size of the feature maps by half."
>
>**[iv]** "The second convolutional layer uses 16 kernels of size 5x5 to produce 16 feature maps of size 10x10."
>
>**[v]** "The third layer is a fully-connected layer with 120 units."
>
>**[vi]** "The output layer has 10 units, one for each class, with a softmax activation function."

<br>

We will also use the following Diagram **[vii]**:

![LeNet-5](https://raw.githubusercontent.com/entbappy/Branching-tutorial/master/lenet/lenet-5.png)

---

## Network architecture

- The network consists of 2 *Convolutional* layers, 2 *Pooling* layers, and 3 *Fully Connected* Layers (**[vii]**).

- *Pooling* is applied after each convolutional layer:
  - C1 → S2 (**[iii]**)
  - C3 → S4 (**[iii]**)

---

## Workflow
We will:
1. Import the necessary layers
2. Demonstrate how the Sequential API works
3. Write the code for the first block
4. Write the code for the second block
5. Write the code for the fully-connected layers
6. Build the model

---

### 1. Imports
**Code:**
>```python
>from tensorflow.keras.models import Sequential
>from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
>```

---

### 2. Sequential API
LeNet-5 can be implemented using the Sequential API, which allows us to stack layers sequentially.

**Code:**
>```python
>model = Sequential()
>```

---

### 3. 1st block
The first block includes the first convolutional layer followed by a pooling layer.

From the paper:

>The first convolutional layer uses **6 kernels** of size **5x5** to produce 6 feature maps of size **28x28** (**[ii]**)

**Code:**

>```python
>model.add(Conv2D(filters=6, kernel_size=5, strides=1, activation='tanh', input_shape=(32, 32, 1)))
>model.add(AveragePooling2D(pool_size=2, strides=2))
>```

---

### 4. 2nd block
The second block includes the second convolutional layer followed by another pooling layer.

From the paper:

>The second convolutional layer uses **16 kernels** of size **5x5** to produce 16 feature maps of size **10x10** (**[iv]**)

**Code:**

>```python
>model.add(Conv2D(filters=16, kernel_size=5, strides=1, activation='tanh'))
>model.add(AveragePooling2D(pool_size=2, strides=2))
>```

---

### 5. Fully Connected Layers
Finally, we flatten the output and pass it through fully-connected layers.

From the paper:

>The first fully-connected layer has **120 units** (**[v]**), followed by another fully-connected layer with **84 units**, and finally, an output layer with **10 units** (**[vi]**).

**Code:**
>```python
>model.add(Flatten())
>model.add(Dense(units=120, activation='tanh'))
>model.add(Dense(units=84, activation='tanh'))
>model.add(Dense(units=10, activation='softmax'))
>```

---

### 6. Model

In order to build the model, we just compile it:

**Code:**
>```python
>model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
>```

---

## Final code

**Code:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense

model = Sequential()

model.add(Conv2D(filters=6, kernel_size=5, strides=1, activation='tanh', input_shape=(32, 32, 1)))
model.add(AveragePooling2D(pool_size=2, strides=2))

model.add(Conv2D(filters=16, kernel_size=5, strides=1, activation='tanh'))
model.add(AveragePooling2D(pool_size=2, strides=2))

model.add(Flatten())
model.add(Dense(units=120, activation='tanh'))
model.add(Dense(units=84, activation='tanh'))
model.add(Dense(units=10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

---

## Model diagram

![LeNet-5](https://raw.githubusercontent.com/entbappy/Branching-tutorial/master/lenet/arch.jpg)


---
