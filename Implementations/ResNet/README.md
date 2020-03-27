# Implementation of ResNet

<br><span style="font-size:14pt;">We will use the tensorflow.keras Functional API to build ResNet</span>
(https://arxiv.org/pdf/1512.03385.pdf)

---

In the paper we can read:

>**[i]** “We adopt batch normalization (BN) [16] right after each convolution and before activation.”
>
>**[ii]** "Donwsampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2."
>
>**[iii]** "(B) The projection shortcut in Eqn.(2) is used to match dimensions (done by 1×1 convolutions). For both options, when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2"
>
>**[iv]** "[...] (B) projection shortcuts are used for increasing dimensions, and other shortcuts are identity;"
>
>**[v]** "The three layers are 1×1, 3×3, and 1×1 convolutions, where the 1×1 layers are responsible for reducing and then increasing (restoring) dimensions, leaving the 3×3 layer a bottleneck with smaller input/output dimensions."
>
>**[vi]** "50-layer ResNet: We replace each 2-layer block in the 34-layer net with this 3-layer bottleneck block, resulting in a 50-layer ResNet (Table 1). We use option B for increasing
dimensions."

<br>

We will also make use of the following Table **[vii]**:

<img src=https://github.com/Machine-Learning-Tokyo/DL-workshop-series/raw/master/Part%20I%20-%20Convolution%20Operations/images/ResNet/ResNet.png width="600">

<br>
<br>

as well the following diagram **[viii]**:
<img src=https://github.com/Machine-Learning-Tokyo/DL-workshop-series/raw/master/Part%20I%20-%20Convolution%20Operations/images/ResNet/ResNet_block.png width="200">

---

## Network architecture

The network starts with a [Conv, BatchNorm, ReLU] block (**[i]**) and continues with a series of **Resnet blocks** (conv*n*.x in **[vii]**) before the final *Avg Pool* and *Fully Connected* layers.

### Resnet block

The *Resnet block* consists of a repetition of blocks similar to the one depicted in **[viii]**. As one can see the input tesnor goes through three Conv-BN-ReLU blocks and the output is added to the input tensor. This type of connection that skips the main body of the block and merges (adds) the input tensor with another one further on is called *skip connection* (right arrow in **[viii]**).

There are two types of skip connections in ResNet: the **Identity** and the **Projection**. In **[viii]** is depicted the **Identity** one. This is used when the input tensor has same shape as the one produced by the last Convolution layer of the block.

However, when the two tensors have different shape, the input tensor must change to get same shape as the other one in order to be able to be added to it. This is done by the **Projection** connection as described in **[iii]** and **[iv]**.

The change in shape happens when we:
- Change the number of filters and thus of feature maps of the output tensor.
This happens at the first sub-block of each *ResNet* block since the output tensor has 4 times the number of feature maps than the input tensor.
- Change the spatial dimensions of the output tensor (downsampling)
which takes place according to **[ii]**.

#### Identity block

The *Identity block* takes a tensor as an input and passes it through 1 stream of:
> 1. a 1x1 *Convolution* layer followed by a *Batch Normalization* and a *Rectified Linear Unit (ReLU)* activation layer
> 2. a 3x3 *Convolution* layer followed by a *Batch Normalization* and a *Rectified Linear Unit (ReLU)* activation layer
> 3. a 1x1 *Convolution* layer followed by a *Batch Normalization* layer
>
> Pay attention at the number of filters (depicted with the letter f at the diagram) which are the same for the first 2 Convolution layer but 4x for the 3rd one.

Then the *output* of this stream is added to the *input* tensor. On the new tensor a *Rectified Linear Unit (ReLU)* activation is applied befor returning it.

<br>

#### Projection block

The *Projection block* takes a tensor as an input and passes it through 2 streams.
- The left stream consists of:
> 1. a 1x1 *Convolution* layer followed by a *Batch Normalization* and a *Rectified Linear Unit (ReLU)* activation layer
> 2. a 3x3 *Convolution* layer followed by a *Batch Normalization* and a *Rectified Linear Unit (ReLU)* activation layer
> 3. a 1x1 *Convolution* layer followed by a *Batch Normalization* layer
>
> Pay attention at the number of filters (depicted with the letter f at the diagram) which are the same for the first 2 Convolution layer but 4x for the 3rd one.


- The right stream consists of:
> a 1x1 *Convolution* layer followed by a *Batch Normalization* layer

The outputs of both streams are then added up to a new tensor on which a *Rectified Linear Unit (ReLU)* activation is applied befor returning it.

<br>

As one can see the only difference between the two blocks is the existence of the Convolution-Batch Normalization sub-block at the right stream.

The reason we need this Convolution layer is:
- To change the number of filters (feature maps) of the tensor after each block.
- To change the size of the tensor after each block.

In order to change the size (downsampling) we use a stride of 2 after specific blocks as described at **[ii]** at the first 1x1 Convolution layer and the Projection's Convolution layer according to **[iii]** and **[v]**.

---

## Workflow
We will:
1. import the neccesary layers
2. write a helper function for the Conv-BatchNorm-ReLU block (**[i]**)
3. write a helper function for the Identity block
4. write a helper function for the Projection block
5. write a helper function for the Resnet block (**[ii]**)
6. use these helper functions to build the model.

---

### 1. Imports
**Code:**
>```python
>from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, \
>      ReLU, Add, MaxPool2D, GlobalAvgPool2D, Dense
>```

---

### 2. *Conv-BatchNorm-ReLU block*
Next, we will build the *Conv-BatchNorm-ReLU block* as a function that will:
- take as inputs:
  - a tensor (**`x`**)
  - the number of filters (**`filters`**)
  - the kernel size (**`kernel_size`**)
  - the strides (**`strides`**)
- run:
    - apply a *Convolution layer* followed by a *Batch Normalization* and a *ReLU* activation
 - return the tensor

**Code:**
> ```python
>def conv_batchnorm_relu(x, filters, kernel_size, strides):
>     x = Conv2D(filters=filters,
>                kernel_size=kernel_size,
>                strides=strides,
>                padding='same')(x)
>     x = BatchNormalization()(x)
>     x = ReLU()(x)
>     return x
> ```

---

### 3. *Identity block*
Now, we will build the *Identity block* as a function that will:
- take as inputs:
  - a tensor (**`tensor`**)
  - the number of filters (**`filters`**)
  - the kernel size (**`kernel_size`**)
  - the strides (**`strides`**)
- run:
    - apply a 1x1 **Conv-BatchNorm-ReLU block** to **`tensor`**
    - apply a 3x3 **Conv-BatchNorm-ReLU block**
    - apply a 1x1 *Convolution layer* with 4 times the filters **`filters`**
    - apply a *Batch normalization*
    - add this tensor with **`tensor`**
    - apply a *ReLU* activation
 - return the tensor

**Code:**
>```python
>def identity_block(tensor, filters):
>     x = conv_batchnorm_relu(tensor, filters=filters, kernel_size=1, strides=1)
>     x = conv_batchnorm_relu(x, filters=filters, kernel_size=3, strides=1)
>     x = Conv2D(filters=4*filters, kernel_size=1, strides=1)(x)  # notice: filters=4*filters
>     x = BatchNormalization()(x)
>
>     x = Add()([x, tensor])
>     x = ReLU()(x)
>     return x
>```

---

### 4. *Projection block*
Now, we will build the *Projection block* which is similar to the *Identity* one.

Remember, this time we need the strides because we want to downsample the tensors at specific blocks according to **[ii]**, **[iii]** and **[v]**:
> “the 1×1 layers are responsible for reducing and then increasing (restoring) dimensions”.

The downsampling at the main stream will take place at the first 1x1 Convolution layer*.
The downsampling at the right stream will take place at its Convolution layer.

**Code:**
> ```python
> def projection_block(tensor, filters, strides):
>     # left stream
>     x = conv_batchnorm_relu(tensor, filters=filters, kernel_size=1, strides=strides) #[v]
>     x = conv_batchnorm_relu(x, filters=filters, kernel_size=3, strides=1)
>     x = Conv2D(filters=4*filters, kernel_size=1, strides=1)(x)  # notice: filters=4*filters
>     x = BatchNormalization()(x)
>
>     # right stream
>     shortcut = Conv2D(filters=4*filters, kernel_size=1, strides=strides)(tensor)  # notice: filters=4*filters
>     shortcut = BatchNormalization()(shortcut)
>
>     x = Add()([x, shortcut])
>     x = ReLU()(x)
>     return x
>```
>
\**Notice that in some implementations downsampling takes place at the 3x3 layer. This is also know as ResNet 1.5 (https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch).*

---

### 5. *Resnet block*
Now that we defined the *Projection block* and the *Identity block* we can use them to define the **Resnet block**.

Based on the **[vii]** (column *50-layer*) for each block we have a number of repetiontions (depicted with *xn* next to the block numbers). The 1st of these blocks will be a *Projection block* and the rest will be *Identity blocks*.

The reason for this is that at the beginning of each block the number of feature maps of the tensor change. Since at the Identity block the input tensor and the output tensor are added, they need to have the same number of feature maps.

Let's build the *Resnet block* as a function that will:
- take as inputs:
  - a tensor (**`x`**)
  - the number of filters (**`filters`**)
  - the total number of repetitions of internal blocks (**`reps`**)
  - the strides (**`strides`**)
- run:
    - apply a projection block with strides: **`strides`**
    - for apply an *Identity block* for $r-1$ times (the $-1$ is because the first block was a *Convolution* one)
- return the tensor

**Code:**
>```python
>def resnet_block(x, filters, reps, strides):
>     x = projection_block(x, filters=filters, strides=strides)
>     for _ in range(reps-1):
>         x = identity_block(x, filters=filters)
>     return x
>```

---

### 6. Model code
Now we are ready to build the model:

**Code:**
>```python
>input = Input(shape=(224, 224, 3))
>
>x = conv_batchnorm_relu(input, filters=64, kernel_size=7, strides=2)  # [3]: 7x7, 64, strides 2
>x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)  # [3]: 3x3 max mool, strides 2
>
>x = resnet_block(x, filters=64, reps=3, strides=1)
>x = resnet_block(x, filters=128, reps=4, strides=2)  # strides=2 ([2]: conv3_1)
>x = resnet_block(x, filters=256, reps=6, strides=2)  # strides=2 ([2]: conv4_1)
>x = resnet_block(x, filters=512, reps=3, strides=2)  # strides=2 ([2]: conv5_1)
>
>x = GlobalAvgPool2D()(x)  # [3]: average pool *it is not written any pool size so we use Global
>
>output = Dense(1000, activation='softmax')(x)  # [3]: 1000-d fc, softmax
>
>from tensorflow.keras import Model
>
>model = Model(input, output)
>```

---

## Final code

**Code:**
```python
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add
from tensorflow.keras.layers import MaxPool2D, GlobalAvgPool2D, Dense


def conv_batchnorm_relu(x, filters, kernel_size, strides):
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def identity_block(tensor, filters):
    x = conv_batchnorm_relu(tensor, filters=filters, kernel_size=1, strides=1)
    x = conv_batchnorm_relu(x, filters=filters, kernel_size=3, strides=1)
    x = Conv2D(filters=4*filters, kernel_size=1, strides=1)(x)  # notice: filters=4*filters
    x = BatchNormalization()(x)

    x = Add()([x, tensor])
    x = ReLU()(x)
    return x


def projection_block(tensor, filters, strides):
    # left stream
    x = conv_batchnorm_relu(tensor, filters=filters, kernel_size=1, strides=strides)
    x = conv_batchnorm_relu(x, filters=filters, kernel_size=3, strides=1)
    x = Conv2D(filters=4*filters, kernel_size=1, strides=1)(x)  # notice: filters=4*filters
    x = BatchNormalization()(x)

    # right stream
    shortcut = Conv2D(filters=4*filters, kernel_size=1, strides=strides)(tensor)  # notice: filters=4*filters
    shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = ReLU()(x)
    return x


def resnet_block(x, filters, reps, strides):
    x = projection_block(x, filters=filters, strides=strides)
    for _ in range(reps-1):  # the -1 is because the first block was a Conv one
        x = identity_block(x, filters=filters)
    return x


input = Input(shape=(224, 224, 3))

x = conv_batchnorm_relu(input, filters=64, kernel_size=7, strides=2)  # [3]: 7x7, 64, strides 2
x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)  # [3]: 3x3 max mool, strides 2

x = resnet_block(x, filters=64, reps=3, strides=1)
x = resnet_block(x, filters=128, reps=4, strides=2)  # s=2 ([2]: conv3_1)
x = resnet_block(x, filters=256, reps=6, strides=2)  # s=2 ([2]: conv4_1)
x = resnet_block(x, filters=512, reps=3, strides=2)  # s=2 ([2]: conv5_1)

x = GlobalAvgPool2D()(x)  # [3]: average pool *it is not written any pool size so we use Global

output = Dense(1000, activation='softmax')(x)  # [3]: 1000-d fc, softmax

from tensorflow.keras import Model

model = Model(input, output)
```

---

## Model diagram

<img src="https://raw.githubusercontent.com/Machine-Learning-Tokyo/CNN-Architectures/master/Implementations/ResNet/ResNet_diagram.svg?sanitize=true">
