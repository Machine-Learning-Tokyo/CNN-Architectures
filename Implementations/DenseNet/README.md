# Implementation of DenseNet

We will use the [tensorflow.keras Functional API](https://www.tensorflow.org/guide/keras/functional) to build DenseNet from the original paper: “[Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)” by Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger.

[Video tutorial](https://www.youtube.com/watch?v=3ZPJyknZolE&list=PLaPdEEY26UXyE3UchW0C742xh542yh0yI&index=8)

---

In the paper we can read:

>**[i]** “Note that each “conv” layer shown in the table corresponds the sequence BN-ReLU-Conv."
>
>**[ii]** "[...] we combine features by concatenating them. Hence, the $\ell th$ layer has $\ell$ inputs, consisting of the feature-maps of all preceding convolutional blocks."
>
>**[iii]** "If each function $H_\ell$ produces $k$ feature-maps, it follows that the $\ell th$ layer has $k_0 + k × (\ell − 1)$ input feature-maps, where $k_0$ is the number of channels in the input layer."
>
>**[iv]** "The initial convolution layer comprises 2k convolutions of size 7×7 with stride 2"
>
>**[v]** "In our experiments, we let each 1×1 convolution produce 4k feature-maps."
>
>**[vi]** "If a dense block contains m feature-maps, we let the following transition layer generate $\lfloor \theta m \rfloor$ output feature-maps, where $0< \theta ≤1$ is referred to as the compression factor. [...] we set $\theta$ = 0.5 in our experiment."

---

We will also make use of the following Table **[vii]** and Diagram **[viii]**:

<img src=https://raw.githubusercontent.com/Machine-Learning-Tokyo/DL-workshop-series/master/Part%20I%20-%20Convolution%20Operations/images/DenseNet/DenseNet.png width="90%">
<img src=https://raw.githubusercontent.com/Machine-Learning-Tokyo/DL-workshop-series/master/Part%20I%20-%20Convolution%20Operations/images/DenseNet/DenseNet_block.png width="60%">

---

## Network architecture

We will implement the Dense-121 (k=32) version of the model (marked with red in **[vii]**).

The model:
- starts with a Convolution-Pooling block
- continues with a series of:
 -- Dense block
 -- Transition layer
- closes with a *Global Average pool* and a *Fully-connected* block.

<br>

In every Dense block the input tensor passes through a series of *conv* operations with fixed number of filters (*k*) and the result of each one is then concatenated to the original tensor **[ii]**. Thus the number of feature maps of the input tensor follows an arithmetic growth at every internal stage of the Dense block by *k* tensors per stage **[iii]**.

In order for the size of the tensor to remain manageable the model makes use of the ***Transition layers***.

At each *Transision layer* the number of feature maps of the input tensor is reduced to half (multiplied by $\theta=0.5$) (**[vi]**).

Also the spatial dimensions of the input tensor are halved by an *Average Pool* layer (**[vii]**).

### Dense block
At each Dense block we have a repetition of:
- 1x1 conv with $4\cdot k$ filters
- 3x3 conv with k filters

blocks.

As it is written in **[i]**: 
>each “conv” layer corresponds the sequence BN-ReLU-Conv

---

## Workflow
We will:
1. import the neccesary layers
2. write the *BN-ReLU-Conv* function (**[i]**)
3. write the *dense_block()* function
4. write the *transition_layer()* function
5. use the functions to build the model 

---

### 1. Imports
**Code:**
>```python
>import tensorflow
>from tensorflow.keras.layers import Input, BatchNormalization, ReLU, \
>      Conv2D, Dense, MaxPool2D, AvgPool2D, GlobalAvgPool2D, Concatenate
>```

---

### 2. BN-ReLU-Conv function
The *BN-ReLU-Conv* function will:
- take as inputs:
    - a tensor (**`x`**)
    - the number of filters for the *Convolution layer* (**`filters`**)
    - the kernel size of the *Convolution layer* (**`kernel_size`**)
- run:
    - apply *Batch Normalization* to `x`
    - apply ReLU to this tensor
    - apply a *Convolution* operation to this tensor
- return the final tensor

**Code:**
>```python
>def bn_rl_conv(x, filters, kernel_size):
>     x = BatchNormalization()(x)
>     x = ReLU()(x)
>     x = Conv2D(filters=filters,
>                kernel_size=kernel_size,
>                padding='same')(x)
>     return x
>```

---

### 3. Dense block

We can use this function to write the *Dense block* function.

This function will:
- take as inputs:
    - a tensor (**`tensor`**)
    - the filters of the conv operations (**`k`**)
    - how many times the conv operations will be applied (**`reps`**)
- run **`reps`** times:
  - apply the 1x1 conv operation with $4\cdot k$ filters (**[v]**)
  - apply the 3x3 conv operation with $k$ filters (**[iii]**)
  - *Concatenate* this tensor with the input **`tensor`**
- return as output the final tensor

**Code:**
>```python
>def dense_block(tensor, k, reps):
>     for _ in range(reps):
>         x = bn_rl_conv(tensor, filters=4*k, kernel_size=1)
>         x = bn_rl_conv(x, filters=k, kernel_size=3)
>         tensor = Concatenate()([tensor, x])
>     return tensor
>```

---


### 4. Transition layer
Following, we will write a function for the transition layer.

This function will:
- take as input:
  - a tensor (**`x`**)
  - the compression factor (**`theta`**)
- run:
  - apply the 1x1 conv operation with **`theta`** times the existing number of filters (**[vi]**)
  - apply Average Pool layer with pool size 2 and stride 2 (**[vii]**)
- return as output the final tensor

Since the number of filters of the input tensor is not known a priori (without computations or hard coded numbers) we can get this number using the `tensorflow.keras.backend.int_shape()` function.
This function returns the shape of a tensor as a tuple of integers

In our case we are interested in the number of feature maps/filters, thus the last number [-1] (channel last mode).

**Code:**
>```python
>def transition_layer(x, theta):
>     f = int(tensorflow.keras.backend.int_shape(x)[-1] * theta)
>     x = bn_rl_conv(x, filters=f, kernel_size=1)
>     x = AvgPool2D(pool_size=2, strides=2, padding='same')(x)
>     return x
>```

---

### 5. Model code
Now that we have defined our helper functions, we can write the code of the model.

The model starts with:
- a Convolution layer with $2\cdot k$ filters, 7x7 kernel size and stride 2 (**[iv]**)
- a 3x3 Max Pool layer with stride 2 (**[vii]**)

and closes with:
- a Global Average pool layer
- a Dense layer with 1000 units and *softmax* activation (**[vii]**)

Notice that after the last *Dense block* there is no *Transition layer*.
For this we use a different letters (d, x) in the `for` loop so that in the end we can take the output of the last *Dense block*.

**Code:**
>```python
>IMG_SHAPE = 224, 224, 3
>k = 32
>theta = 0.5
>repetitions = 6, 12, 24, 16
>
>input = Input(IMG_SHAPE)
>
>x = Conv2D(2*k, 7, strides=2, padding='same')(input)
>x = MaxPool2D(3, strides=2, padding='same')(x)
>
>for reps in repetitions:
>     d = dense_block(x, k, reps)
>     x = transition_layer(d, theta)
>
>x = GlobalAvgPool2D()(d)
>
>output = Dense(1000, activation='softmax')(x)
>
>from tensorflow.keras import Model 
>model = Model(input, output)
>```

---

## Final code

**Code:**
```python
import tensorflow
from tensorflow.keras.layers import Input, BatchNormalization, ReLU, \
     Conv2D, Dense, MaxPool2D, AvgPool2D, GlobalAvgPool2D, Concatenate


def bn_rl_conv(x, filters, kernel_size):
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               padding='same')(x)
    return x


def dense_block(tensor, k, reps):
    for _ in range(reps):
        x = bn_rl_conv(tensor, filters=4*k, kernel_size=1)
        x = bn_rl_conv(x, filters=k, kernel_size=3)
        tensor = Concatenate()([tensor, x])
    return tensor


def transition_layer(x, theta):
    f = int(tensorflow.keras.backend.int_shape(x)[-1] * theta)
    x = bn_rl_conv(x, filters=f, kernel_size=1)
    x = AvgPool2D(pool_size=2, strides=2, padding='same')(x)
    return x


IMG_SHAPE = 224, 224, 3
k = 32
theta = 0.5
repetitions = 6, 12, 24, 16

input = Input(IMG_SHAPE)

x = Conv2D(2*k, 7, strides=2, padding='same')(input)
x = MaxPool2D(3, strides=2, padding='same')(x)

for reps in repetitions:
    d = dense_block(x, k, reps)
    x = transition_layer(d, theta)

x = GlobalAvgPool2D()(d)

output = Dense(1000, activation='softmax')(x)

from tensorflow.keras import Model 
model = Model(input, output)
```

---

## Model diagram

<img src="https://raw.githubusercontent.com/Machine-Learning-Tokyo/CNN-Architectures/master/Implementations/DenseNet/DenseNet_diagram.svg?sanitize=true">
