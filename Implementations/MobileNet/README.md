# Implementation of MobileNet

<br><span style="font-size:14pt;">We will use the tensorflow.keras Functional API to build MobileNet</span>
(https://arxiv.org/pdf/1704.04861.pdf)

---

In the paper we can read:

>**[i]** “All layers are followed by a batchnorm and ReLU nonlinearity”.

<br>


We will also make use of the following Table **[ii]**:

<img src=https://raw.githubusercontent.com/Machine-Learning-Tokyo/DL-workshop-series/master/Part%20I%20-%20Convolution%20Operations/images/MobileNet/MobileNet.png width="500">

<br>

as well the following Diagram **[iii]** of the *Mobilenet block*:

<img src=https://raw.githubusercontent.com/Machine-Learning-Tokyo/DL-workshop-series/master/Part%20I%20-%20Convolution%20Operations/images/MobileNet/MobileNet_block.png width="150">

## Network architecture

The network starts with a (Conv, BatchNorm, ReLU) block and continues with a series of **Mobilenet blocks** before the final *Avg Pool* and *Fully Connected* layers.

<br/>


### Mobilenet block

The *Mobilenet block* is depicted at the bottom right figure of the above image. Specifically it consists of 6 layers:

1. a 3x3 *Depth Wise Convolution* layer
2. a *Batch Normalization* layer (**[i]**)
3. a *Rectified Linear Unit (ReLU)* activation layer (**[i]**)
4. a 1x1 *Convolutional* layer
5. a *Batch Normalization* layer
6. a *Rectified Linear Unit (ReLU)* activation layer

---

## Workflow
We will:
1. import the neccesary layers
2. write a helper function for the MobileNet block **[iii]**
3. build the stem of the model
4. use these helper functions to build main part of the model.

---

### 1. Imports
**Code:**
>```python
>from tensorflow.keras.layers import Input, DepthwiseConv2D, \
>      Conv2D, BatchNormalization, ReLU, AvgPool2D, Flatten, Dense
>```

---

### 2. MobileNet block
Next, we will build the *Mobilenet block* as a function that will
- take as input:
  - a tensor (**`x`**)
  - the number of filters for the Convolutional layer (**`filters`**)
  - the strides for the Depthwise Convolutional layer (**`strides`**)
- run:
    - apply a 3x3 *Depthwise Convolution layer* with **`strides`** strides followed by a *Batch Normalization* and a *ReLU* activation
    - apply a 1x1 *Convolution layer* with **`filters`** filters followed by a *Batch Normalization* and a *ReLU* activation
- return the tensor **`output`**.

**Code:**
>```python
>def mobilenet_block(x, filters, strides):
>     x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same')(x)
>     x = BatchNormalization()(x)
>     x = ReLU()(x)
>
>     x = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same')(x)
>     x = BatchNormalization()(x)
>     x = ReLU()(x)
>     return x
>```

<br>

At **[ii]** one can see the following pattern after the first Convolution layer:
- Conv dw/s1
- Conv/s1

These two lines consist one Mobilenet block.
- the number after the *s* of the first line is the strides of the Depthwise layer
- the last number of the *Filter Shape* column of the second line is the number of filters of the Convolution layer

<br/>

For example, this is the first Mobilenet block of **[ii]**:

| Type / Stride    	| Filter Shape 	|
|---------------   	|:------------:	|
| Conv dw/s**1**   	| 3x3x32 dw    	|
| Conv/s1           | 1x1x32x**64** |

The corresponding call of the mobilenet_block() function would be:

>`mobilenet_block(x, fitlers=64, strides=1)`

---

### 3. Stem of the model

From **[ii]**:

| Type / Stride 	| Filter Shape 	|
|---------------	|:------------:	|
| Conv/s2      	  | 3x3x3x32    	|


**Code:**
>```python
>INPUT_SHAPE = 224, 224, 3
>
>input = Input(INPUT_SHAPE)
>x = Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(input)
>x = BatchNormalization()(x)
>x = ReLU()(x)
>```

---

### 4. Main part of the model

From **[ii]**:


| Type / Stride 	| Filter Shape 	|
|---------------	|:------------:	|
| Conv dw/s1    	| 3x3x32 dw    	|
| Conv/s1       	| 1x1x32x64    	|

**Code:**
>```python
>x = mobilenet_block(x, filters=64, strides=1)
>```

---



From **[ii]**:

| Type / Stride 	| Filter Shape 	|
|---------------	|:------------:	|
| Conv dw/s2    	| 3x3x64 dw    	|
| Conv/s1       	| 1x1x64x128   	|
| Conv dw/s1    	| 3x3x128 dw   	|
| Conv/s1       	| 1x1x128x128  	|

**Code:**
>```python
>x = mobilenet_block(x, filters=128, strides=2)
>x = mobilenet_block(x, filters=128, strides=1)
>```

---


From **[ii]**:

| Type / Stride 	| Filter Shape 	|
|---------------	|:------------:	|
| Conv dw/s2    	| 3x3x128 dw   	|
| Conv/s1       	| 1x1x128x256  	|
| Conv dw/s1    	| 3x3x256 dw   	|
| Conv/s1       	| 1x1x256x256  	|

**Code:**
>```python
>x = mobilenet_block(x, filters=256, strides=2)
>x = mobilenet_block(x, filters=256, strides=1)
>```

---


From **[ii]**:

| Type / Stride       	 	| Filter Shape         		 			|
|--------------:       		| :-----------:       		 		 	|
| Conv dw/s2       	    	| 3x3x256 dw       	   	        |
| Conv/s1       	       	| 1x1x256x512       	        	|
| 5x Conv dw/s1<br>Conv/s1| 3x3x512 dw<br>1x1x512x512   	|

**Code:**
>```python
>x = mobilenet_block(x, filters=512, strides=2)
>for _ in range(5):
>     x = mobilenet_block(x, filters=512, strides=1)
>```

---


From **[ii]**:

| Type / Stride 	| Filter Shape  	|
|---------------	|--------------:	|
| Conv dw/s2    	| 3x3x512 dw    	|
| Conv/s1       	| 1x1x512x1024  	|
| Conv dw/s1    	| 3x3x1024 dw   	|
| Conv/s1       	| 1x1x1024x1024 	|
| Avg Pool/s1   	| Pool 7x7      	|
| FC/s1         	| 1024x1000     	|
| Softmax/s1    	| Classifier    	|

**Code:**
>```python
>x = mobilenet_block(x, filters=1024, strides=2)
>x = mobilenet_block(x, filters=1024, strides=1)
>
>x = AvgPool2D(pool_size=7, strides=1)(x)
>output = Dense(units=1000, activation='softmax')(x)
>```

---


## Final code

**Code:**
```python
from tensorflow.keras.layers import Input, DepthwiseConv2D, \
     Conv2D, BatchNormalization, ReLU, AvgPool2D, Flatten, Dense

def mobilenet_block(x, filters, strides):
    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

INPUT_SHAPE = 224, 224, 3

input = Input(INPUT_SHAPE)
x = Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(input)
x = BatchNormalization()(x)
x = ReLU()(x)

x = mobilenet_block(x, filters=64, strides=1)

x = mobilenet_block(x, filters=128, strides=2)
x = mobilenet_block(x, filters=128, strides=1)

x = mobilenet_block(x, filters=256, strides=2)
x = mobilenet_block(x, filters=256, strides=1)

x = mobilenet_block(x, filters=512, strides=2)
for _ in range(5):
    x = mobilenet_block(x, filters=512, strides=1)
  
x = mobilenet_block(x, filters=1024, strides=2)
x = mobilenet_block(x, filters=1024, strides=1)

x = AvgPool2D(pool_size=7, strides=1)(x)
output = Dense(units=1000, activation='softmax')(x)

from tensorflow.keras import Model

model = Model(inputs=input, outputs=output)
```

---

## Model diagram

<img src="https://raw.githubusercontent.com/Machine-Learning-Tokyo/CNN-Architectures/master/Implementations/MobileNet/MobileNet_diagram.svg?sanitize=true">
