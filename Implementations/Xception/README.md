# Implementation of Xception

We will use the [tensorflow.keras Functional API](https://www.tensorflow.org/guide/keras/functional) to build Xception from the original paper: “[Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)” by François Chollet.

[Video tutorial](https://www.youtube.com/watch?v=nMBCSroJ7bY&list=PLaPdEEY26UXyE3UchW0C742xh542yh0yI&index=6)

---

In the paper we can read:

>**[i]** “all Convolution and SeparableConvolution layers are followed by batch normalization [7] (not included in the diagram)."
>
>**[ii]** "All SeparableConvolution layers use a depth multiplier of 1 (no depth expansion)."

<br>

We will also use the following Diagram **[iii]**:

<img src=https://raw.githubusercontent.com/Machine-Learning-Tokyo/DL-workshop-series/master/Part%20I%20-%20Convolution%20Operations/images/Xception/Xception.png width="600">

<br>

as well the following Table **[iv]** to check the total number of parameters:

<img src=https://raw.githubusercontent.com/Machine-Learning-Tokyo/DL-workshop-series/master/Part%20I%20-%20Convolution%20Operations/images/Xception/Xception_parameters.png width="200">

---

## Network architecture

The model is separated in 3 flows as depicted at **[iii]**:
- Entry flow
- Middle flow with 8 repetitions of the same block
- Exit flow

According to **[i]** all Convolution and Separable Convolution layers are followed by batch normalization.

---

## Workflow
We will:
1. import the neccesary layers
2. write one helper function for the Conv-BatchNorm block and one for the SeparableConv-BatchNorm block according to **[i]**
3. write one function for each one of the 3 flows according to **[iii]**
4. use these helper functions to build the model.

---

### 1. Imports
**Code:**
>```python
>from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, \
>      Add, Dense, BatchNormalization, ReLU, MaxPool2D, GlobalAvgPool2D
>```

---

### 2.1. Conv-BatchNorm block
The *Conv-BatchNorm block* will:
- take as inputs:
  - a tensor (**`x`**)
  - the number of filters of the *Convolution layer* (**`filters`**)
  - the kernel size of the *Convolution layer* (**`kernel_size`**)
  - the strides of the *Convolution layer* (**`strides`**)
- run:
  - apply a *Convolution layer* to **`x`**
  - apply a *Batch Normalization* layer to this tensor
- return the tensor

**Code:**
>```python
>def conv_bn(x, filters, kernel_size, strides=1):
>     x = Conv2D(filters=filters,
>                kernel_size=kernel_size,
>                strides=strides,
>                padding='same',
>                use_bias=False)(x)
>     x = BatchNormalization()(x)
>     return x
>```

***Note***: We include *use_bias=False* for the final number of parameters to match the ones written at **[iv]**.

---

### 2.2. SeparableConv-BatchNorm
The *SeparableConv-BatchNorm block* has similar structure with the *Conv-BatchNorm* one

**Code:**
>```python
>def sep_bn(x, filters, kernel_size, strides=1):
>     x = SeparableConv2D(filters=filters,
>                         kernel_size=kernel_size,
>                         strides=strides,
>                         padding='same',
>                         use_bias=False)(x)
>     x = BatchNormalization()(x)
>     return x
>```

---

### 3.1. Entry flow
<img src=https://raw.githubusercontent.com/Machine-Learning-Tokyo/DL-workshop-series/master/Part%20I%20-%20Convolution%20Operations/images/Xception/entry_flow.png width="300">

**Code:**
>```python
>def entry_flow(x):
>     x = conv_bn(x, filters=32, kernel_size=3, strides=2)
>     x = ReLU()(x)
>     x = conv_bn(x, filters=64, kernel_size=3)
>     tensor = ReLU()(x)
>   
>     x = sep_bn(tensor, filters=128, kernel_size=3)
>     x = ReLU()(x)
>     x = sep_bn(x, filters=128, kernel_size=3)
>     x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
>
>     tensor = conv_bn(tensor, filters=128, kernel_size=1, strides=2)
>
>     x = Add()([tensor, x])
>     x = ReLU()(x)
>     x = sep_bn(x, filters=256, kernel_size=3)
>     x = ReLU()(x)
>     x = sep_bn(x, filters=256, kernel_size=3)
>     x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
>
>     tensor = conv_bn(tensor, filters=256, kernel_size=1, strides=2)
>
>     x = Add()([tensor, x])
>     x = ReLU()(x)
>     x = sep_bn(x, filters=728, kernel_size=3)
>     x = ReLU()(x)
>     x = sep_bn(x, filters=728, kernel_size=3)
>     x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
>
>     tensor = conv_bn(tensor, filters=728, kernel_size=1, strides=2)
>     x = Add()([tensor, x])
>
>     return x
>```

---

### 3.2. Middle flow
<img src=https://raw.githubusercontent.com/Machine-Learning-Tokyo/DL-workshop-series/master/Part%20I%20-%20Convolution%20Operations/images/Xception/middle_flow.png width="250">

**Code:**
>```python
>def middle_flow(tensor):
>     for _ in range(8):
>         x = ReLU()(tensor)
>         x = sep_bn(x, filters=728, kernel_size=3)
>         x = ReLU()(x)
>         x = sep_bn(x, filters=728, kernel_size=3)
>         x = ReLU()(x)
>         x = sep_bn(x, filters=728, kernel_size=3)
>
>         tensor = Add()([tensor, x])
>
>     return tensor
>```

---

### 3.3. Exit flow
<img src=https://raw.githubusercontent.com/Machine-Learning-Tokyo/DL-workshop-series/master/Part%20I%20-%20Convolution%20Operations/images/Xception/exit_flow.png width="300">

**Code:**
>```python
>def exit_flow(tensor):
>     x = ReLU()(tensor)
>     x = sep_bn(x, filters=728, kernel_size=3)
>     x = ReLU()(x)
>     x = sep_bn(x, filters=1024, kernel_size=3)
>     x = MaxPool2D(3, strides=2, padding='same')(x)
>
>     tensor = conv_bn(tensor, filters=1024, kernel_size=1, strides=2)
>
>     x = Add()([tensor, x])
>     x = sep_bn(x, filters=1536, kernel_size=3)
>     x = ReLU()(x)
>     x = sep_bn(x, filters=2048, kernel_size=3)
>     x = ReLU()(x)
>     x = GlobalAvgPool2D()(x)
>     x = Dense(units=1000, activation='softmax')(x)
>
>     return x
>```

---


### 4. Model code
**Code:**
>```python
>input = Input(shape=[299, 299, 3])
>
>x = entry_flow(input)
>x = middle_flow(x)
>output = exit_flow(x)
>
>from tensorflow.keras import Model 
>model = Model(input, output)
>```

---

### Check number of parameters

We can also check the total number of trainable parameters of the model by calling `count_params()` on each result element of `model.trainable_weights`.

According to **[iv]** there are 22,855,952 trainable parameters at Xception model.

**Code:**
```python
>>> import numpy as np
>>> import tensorflow.keras.backend as K
>>> np.sum([K.count_params(p) for p in model.trainable_weights])
22855952
```

---

## Final code

**Code:**
```python
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, \
     Add, Dense, BatchNormalization, ReLU, MaxPool2D, GlobalAvgPool2D

def conv_bn(x, filters, kernel_size, strides=1):
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding='same',
               use_bias=False)(x)
    x = BatchNormalization()(x)
    return x


def sep_bn(x, filters, kernel_size, strides=1):
    x = SeparableConv2D(filters=filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding='same',
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    return x


def entry_flow(x):
    x = conv_bn(x, filters=32, kernel_size=3, strides=2)
    x = ReLU()(x)
    x = conv_bn(x, filters=64, kernel_size=3)
    tensor = ReLU()(x)

    x = sep_bn(tensor, filters=128, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters=128, kernel_size=3)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    tensor = conv_bn(tensor, filters=128, kernel_size=1, strides=2)

    x = Add()([tensor, x])
    x = ReLU()(x)
    x = sep_bn(x, filters=256, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters=256, kernel_size=3)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    tensor = conv_bn(tensor, filters=256, kernel_size=1, strides=2)

    x = Add()([tensor, x])
    x = ReLU()(x)
    x = sep_bn(x, filters=728, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters=728, kernel_size=3)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    tensor = conv_bn(tensor, filters=728, kernel_size=1, strides=2)
    x = Add()([tensor, x])

    return x


def middle_flow(tensor):
    for _ in range(8):
        x = ReLU()(tensor)
        x = sep_bn(x, filters=728, kernel_size=3)
        x = ReLU()(x)
        x = sep_bn(x, filters=728, kernel_size=3)
        x = ReLU()(x)
        x = sep_bn(x, filters=728, kernel_size=3)

        tensor = Add()([tensor, x])

    return tensor


def exit_flow(tensor):
    x = ReLU()(tensor)
    x = sep_bn(x, filters=728, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters=1024, kernel_size=3)
    x = MaxPool2D(3, strides=2, padding='same')(x)

    tensor = conv_bn(tensor, filters=1024, kernel_size=1, strides=2)

    x = Add()([tensor, x])
    x = sep_bn(x, filters=1536, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters=2048, kernel_size=3)
    x = ReLU()(x)
    x = GlobalAvgPool2D()(x)
    x = Dense(units=1000, activation='softmax')(x)

    return x


input = Input(shape=[299, 299, 3])

x = entry_flow(input)
x = middle_flow(x)
output = exit_flow(x)

from tensorflow.keras import Model 
model = Model(input, output)
```

---

## Model diagram

<img src="https://raw.githubusercontent.com/Machine-Learning-Tokyo/CNN-Architectures/master/Implementations/Xception/Xception_diagram.svg?sanitize=true">
