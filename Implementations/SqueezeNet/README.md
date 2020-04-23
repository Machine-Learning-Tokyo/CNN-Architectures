# Implementation of SqueezeNet


We will use the [tensorflow.keras Functional API](https://www.tensorflow.org/guide/keras/functional) to build SqueezeNet from the original paper: “[SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360)” by Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, Kurt Keutzer.

[Video tutorial](https://www.youtube.com/watch?v=W4UbinapGMY&list=PLaPdEEY26UXyE3UchW0C742xh542yh0yI&index=7)

---

In the paper we can read:

>**[i]** “[...] we implement our expand layer with two separate convolution layers: a layer with 1x1 filters, and a layer with 3x3 filters. Then, we concatenate the outputs of these layers together in the channel dimension."

<br>

We will also make use of the following Table **[ii]**:

<img src=https://raw.githubusercontent.com/Machine-Learning-Tokyo/DL-workshop-series/master/Part%20I%20-%20Convolution%20Operations/images/SqueezeNet/SqueezeNet.png? width="600">

as well the following Diagrams **[iii]** and **[iv]**

<img src=https://raw.githubusercontent.com/Machine-Learning-Tokyo/DL-workshop-series/master/Part%20I%20-%20Convolution%20Operations/images/SqueezeNet/SqueezeNet_diagram.png? width="350">

<img src=https://raw.githubusercontent.com/Machine-Learning-Tokyo/DL-workshop-series/master/Part%20I%20-%20Convolution%20Operations/images/SqueezeNet/SqueezeNet_diagram_2.png? width="70">

---

## Network architecture

Based on **[ii]** the network 
- starts with a Convolution-MaxPool block 
- continues with a series of **Fire blocks** separated by MaxPool layers 
- finishes with *Convolution* and *Average Pool* layers.

Notice that there is no *Fully Connected* layer in the model which means that the network can process different image sizes.


### Fire block

The *Fire block* is depicted at **[iii]** and consists of:
>1. a 1x1 *Convolution* layer that outputs the `squeezed` tensor
>2. a 1x1 *Convolution* layer and a 3x3 *Convolution* layer applied on the *squeeze* tensor and the ouputs of which are then concatenated as described in **[i]**

---

## Workflow
We will:
1. import the neccesary layers
2. write a helper function for the Fire block (**[iii]**)
3. write the stem of the model
4. use the helper function to write the main part of the model
5. write the last part of the model

---

### 1. Imports
**Code:**
>```python
>from tensorflow.keras.layers import Input, Conv2D, Concatenate, \
>      MaxPool2D, GlobalAvgPool2D, Activation
>```

---

### 2. Fire block
Next, we will write the Fire block function

This function will:
- take as inputs:
  - a tensor (**`x`**)
  - the filters of the 1st 1x1 Convolution layer (**`squeeze_filters`**)
  - the filters of the 2nd 1x1 Convolution and the 3x3 Convolution layers (**`expand_filters`**)
- run:
  - apply a 1x1 conv operation on **`x`** to get **`squeezed`** tensor
  - apply a 1x1 conv and a 3x3 conv operation on **`squeezed`**
  - *Concatenate* these two tensors
- return the concatenated tensor

**Code:**
>```python
>def fire_block(x, squeeze_filters, expand_filters):
>     squeezed = Conv2D(filters=squeeze_filters,
>                       kernel_size=1,
>                       activation='relu')(x)
>     expanded_1x1 = Conv2D(filters=expand_filters,
>                         kernel_size=1,
>                         activation='relu')(squeezed)
>     expanded_3x3 = Conv2D(filters=expand_filters,
>                         kernel_size=3,
>                         padding='same',
>                         activation='relu')(squeezed)
>
>     output = Concatenate()([expanded_1x1, expanded_3x3])
>     return output
>```

---

### 3. Model stem
Based on **[ii]**:

| layer name/type 	| output size 	| filter size / stride 	|
|-----------------	|:-----------:	|--------------------:	|
| input image     	| 224x224x3   	|                      	|
| conv1           	| 111x111x96  	| 7x7/2 (x96)          	|
| maxpool1        	| 55x55x96    	| 3x3/2                	|

the model starts with:
>1. a Convolution layer with 96 filters and kernel size 7x7 applied on a 224x224x3 input image
>2. a MaxPool layer with pool size 3x3 and stride 2

**Code:**
>```python
>input = Input([224, 224, 3])
>
>x = Conv2D(96, 7, strides=2, padding='same', activation='relu')(input)
>x = MaxPool2D(3, strides=2, padding='same')(x)
>```

---

### 4. Main part
Based on **[ii]**:

| layer name/type 	| filter size / stride 	| s1x1(#1x1 squeeze) 	| e1x1(#1x1 expand) 	| e3x3(#3x3 expand) 	|
|-----------------	|----------------------	|--------------------	|-------------------	|-------------------	|
| fire2           	|                      	| 16                 	| 64                	| 64                	|
| fire3           	|                      	| 16                 	| 64                	| 64                	|
| fire4           	|                      	| 32                 	| 128               	| 128               	|
| maxpool4        	| 3x3/2                	|                    	|                   	|                   	|
| fire5           	|                      	| 32                 	| 128               	| 128               	|
| fire6           	|                      	| 48                 	| 192               	| 192               	|
| fire7           	|                      	| 48                 	| 192               	| 192               	|
| fire8           	|                      	| 64                 	| 256               	| 256               	|
| maxpool8        	| 3x3/2                	|                    	|                   	|                   	|
| fire9           	|                      	| 64                 	| 256               	| 256               	|

the model continues with:
>1. Fire block (fire2) with 16 squeeze and 64 expand filters
>2. Fire block (fire3) with 16 squeeze and 64 expand filters
>3. Fire block (fire4) with 32 squeeze and 128 expand filters
>4. a MaxPool layer (maxpool4) with pool size 3x3 and stride 2
>1. Fire block (fire5) with 32 squeeze and 128 expand filters
>2. Fire block (fire6) with 48 squeeze and 192 expand filters
>3. Fire block (fire7) with 48 squeeze and 192 expand filters
>3. Fire block (fire8) with 64 squeeze and 256 expand filters
>4. a MaxPool layer (maxpool8) with pool size 3x3 and stride 2
>3. Fire block (fire9) with 64 squeeze and 256 expand filters

**Code:**
>```python
>x = fire_block(x, squeeze_filters=16, expand_filters=64)
>x = fire_block(x, squeeze_filters=16, expand_filters=64)
>x = fire_block(x, squeeze_filters=32, expand_filters=128)
>x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
>
>x = fire_block(x, squeeze_filters=32, expand_filters=128)
>x = fire_block(x, squeeze_filters=48, expand_filters=192)
>x = fire_block(x, squeeze_filters=48, expand_filters=192)
>x = fire_block(x, squeeze_filters=64, expand_filters=256)
>x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
>
>x = fire_block(x, squeeze_filters=64, expand_filters=256)
>```

---

### 5. Last part
Based on **[ii]**:

| layer name/type 	| filter size / stride 	|
|-----------------	|----------------------	|
| conv10          	| 1x1/1 (x1000)        	|
| avgpool10       	| 13x13/1              	|

the model ends with:
>1. a Convolution layer with 1000 filters and kernel size 1x1
>2. a Average Pool layer with stride 1 which based on **[iv]** is *Global*
>3. a *Softmax* activation applied on the output number (**[iv]**)

**Code:**
>```python
>x = Conv2D(filters=1000, kernel_size=1)(x)
>x = GlobalAvgPool2D()(x)
>
>output = Activation('softmax')(x)
>
>from tensorflow.keras import Model
>model = Model(input, output)
>```


### Check number of parameters

We can also check the total number of parameters of the model by calling `count_params()` on each result element of `model.trainable_weights`.

According to **[ii]** (col: #parameter before pruning) there are 1,248,424 (total) parameters at SqueezeNet model.

**Code:**
```python
>>> import numpy as np
>>> import tensorflow.keras.backend as K
>>> int(np.sum([K.count_params(p) for p in model.trainable_weights]))
1248424
```

---

## Final code

**Code:**
```python
from tensorflow.keras.layers import Input, Conv2D, Concatenate, \
     MaxPool2D, GlobalAvgPool2D, Activation


def fire_block(x, squeeze_filters, expand_filters):
    squeezed = Conv2D(filters=squeeze_filters,
                      kernel_size=1,
                      activation='relu')(x)
    expanded_1x1 = Conv2D(filters=expand_filters,
                        kernel_size=1,
                        activation='relu')(squeezed)
    expanded_3x3 = Conv2D(filters=expand_filters,
                        kernel_size=3,
                        padding='same',
                        activation='relu')(squeezed)

    output = Concatenate()([expanded_1x1, expanded_3x3])
    return output


input = Input([224, 224, 3])

x = Conv2D(96, 7, strides=2, padding='same', activation='relu')(input)
x = MaxPool2D(3, strides=2, padding='same')(x)


x = fire_block(x, squeeze_filters=16, expand_filters=64)
x = fire_block(x, squeeze_filters=16, expand_filters=64)
x = fire_block(x, squeeze_filters=32, expand_filters=128)
x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

x = fire_block(x, squeeze_filters=32, expand_filters=128)
x = fire_block(x, squeeze_filters=48, expand_filters=192)
x = fire_block(x, squeeze_filters=48, expand_filters=192)
x = fire_block(x, squeeze_filters=64, expand_filters=256)
x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

x = fire_block(x, squeeze_filters=64, expand_filters=256)


x = Conv2D(filters=1000, kernel_size=1)(x)
x = GlobalAvgPool2D()(x)

output = Activation('softmax')(x)

from tensorflow.keras import Model
model = Model(input, output)
```

---

## Model diagram

<img src="https://raw.githubusercontent.com/Machine-Learning-Tokyo/CNN-Architectures/master/Implementations/SqueezeNet/SqueezeNet_diagram.svg?sanitize=true">
