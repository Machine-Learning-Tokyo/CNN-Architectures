# Implementation of ShuffleNet

We will use the [tensorflow.keras Functional API](https://www.tensorflow.org/guide/keras/functional) to build ShuffleNet from the original paper: “[ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)” by Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, Jian Sun.

[Video tutorial](https://www.youtube.com/watch?v=lWMd_wJMeOE&list=PLaPdEEY26UXyE3UchW0C742xh542yh0yI&index=9)

---

In the paper we can read:

>**[i]** “The first building block in each stage is applied with stride = 2. Other hyper-parameters within a stage stay the same, and for the next stage the output channels are doubled”.
>
>**[ii]** “Similar to [9], we set the number of bottleneck channels to 1/4 of the output channels for each ShuffleNet unit"
>
>**[iii]** "we add a Batch Normalization layer [15] after each of the convolutions to make end-to-end training easier."
>
>**[iv]** "Note that for Stage 2, we do not apply group convolution on the first pointwise layer because the number of input channels is relatively small."

<br>

We will also make use of the following Table **[v]**:

<img src=https://raw.githubusercontent.com/Machine-Learning-Tokyo/DL-workshop-series/master/Part%20I%20-%20Convolution%20Operations/images/ShuffleNet/ShuffleNet.png width="600">

<br>

as well the following Diagrams **[vi]**

<img src=https://raw.githubusercontent.com/Machine-Learning-Tokyo/DL-workshop-series/master/Part%20I%20-%20Convolution%20Operations/images/ShuffleNet/ShuffleNet_diagram_1.png width="600">

<sub>Figure 2. ShuffleNet Units. a) bottleneck unit [9] with depthwise convolution (DWConv) [3, 12]; b) ShuffleNet unit with pointwise group convolution (GConv) and channel shuffle; c) ShuffleNet unit with stride = 2.</sub>

 and **[vii]**
<img src=https://raw.githubusercontent.com/Machine-Learning-Tokyo/DL-workshop-series/master/Part%20I%20-%20Convolution%20Operations/images/ShuffleNet/ShuffleNet_diagram_2.png width="600">

<sub>Figure 1. Channel shuffle with two stacked group convolutions. GConv stands for group convolution. a) two stacked convolution layers with the same number of groups. Each output channel only relates to the input channels within the group. No cross talk; b) input and output channels are fully related when GConv2 takes data from different groups after GConv1; c) an equivalent implementation to b) using channel shuffle.</sub>

---

## Network architecture
Based on **[v]** the model starts with a stem of Convolution-Max Pool and continues with a number of **Stages** before the final Global Pool-Fully Connected layers.

Each **Stage** consists of two parts:
1. One **Shufflenet block** with strides 2 **[vi.b]**
2. a number of repeated **Shufflenet blocks** with strides 1 **[vi.c]**

Each one of the right most columns of **[v]** corresponds to a model architecture with different number of internal groups (g). In our case we are going to implement the "*g = 8*" model, however the code will be general enough to support any other combination of number of:
- groups
- stages
- repetitions per stage

### Shufflenet block
The Shufflenet block is the building block of this network. Similar to the ResNet block there are two variations of the block based on whether the spatial dimensions of the input tensor change (strides = 2) or not (strides = 1).

In the first case we apply a 3x3 Average Pool with strides 2 at the shortcut connection as depicted at **[vi.c]**.

The main branch of the block consists of:
1. 1x1 **Group Convolution** with 1/4 filters (GConv) followed by Batch Normalization and ReLU (**[ii]**)
2. **Channel Shuffle** operation
3. 3x3 DepthWise Convolution (with or w/o strides=2) followed by Batch Normalizaion
4. 1x1 **Group Convolution** followed by Batch Normalizaion

The tensors of the main branch and the shortcut connection are then concatenated and a ReLU activation is applied to the output.

### Group Convolution
The idea of *Group Convolution* is to separate the input tensor to g sub-tensors each one with $1/g$ distinct channels of the initial tesnsor. Then we apply a 1x1 Convolution to each sub-tensor and finally we concatenate all the sub-tensors together (**[vii]**).


### Channel Shuffle
Channel shuffle is an operation of shuffling the channels of the input tensor as shown at **[vii.b,c]**.
In order to shuffle the channels we
1. reshape the input tensor:
>from: `width x height x channels`
>
>to: `width x height x groups x (channels/groups)`

2. prermute the last two dimensions
3. reshape the tensor to the original shape

A simple example of the results of this operation can be seen at the following application of the operation on a 6-element array

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{matrix}&space;1&space;&&space;2&space;&&space;3&space;&&space;4&space;&&space;5&space;&&space;6&space;\end{matrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{matrix}&space;1&space;&&space;2&space;&&space;3&space;&&space;4&space;&&space;5&space;&&space;6&space;\end{matrix}" title="\begin{matrix} 1 & 2 & 3 & 4 & 5 & 6 \end{matrix}" /></a>

1. reshape to groups x (n / groups) (groups=2)
<br>
<br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{matrix}&space;1&space;&&space;2&space;&&space;3\\&space;4&space;&&space;5&space;&&space;6&space;\end{matrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{matrix}&space;1&space;&&space;2&space;&&space;3\\&space;4&space;&&space;5&space;&&space;6&space;\end{matrix}" title="\begin{matrix} 1 & 2 & 3\\ 4 & 5 & 6 \end{matrix}" /></a>
<br>
2. prermute the dimensions

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{matrix}&space;1&space;&&space;4\\&space;2&space;&&space;5\\&space;3&space;&&space;6&space;\end{matrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{matrix}&space;1&space;&&space;4\\&space;2&space;&&space;5\\&space;3&space;&&space;6&space;\end{matrix}" title="\begin{matrix} 1 & 4\\ 2 & 5\\ 3 & 6 \end{matrix}" /></a>

<br>
3. reshape to the original shape
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{matrix}&space;1&space;&&space;4&space;&&space;2&space;&&space;5&space;&&space;3&space;&&space;6&space;\end{matrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{matrix}&space;1&space;&&space;4&space;&&space;2&space;&&space;5&space;&&space;3&space;&&space;6&space;\end{matrix}" title="\begin{matrix} 1 & 4 & 2 & 5 & 3 & 6 \end{matrix}" /></a>

---

## Workflow
We will:
1. import the neccesary layers
2. write a helper function for the **Stage**
3. write a helper function for the **Shufflenet block**
4. write a helper function for the **Group Convolution**
5. write a helper function for the **Channel Shuffle**
6. write the stem of the model
7. use the helper function to write the main part of the model
8. write the last part of the model and build it

---

### 1. Imports
**Code:**
>```python
>from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, \
>      Dense, Concatenate, Add, ReLU, BatchNormalization, AvgPool2D, \
>      MaxPool2D, GlobalAvgPool2D, Reshape, Permute, Lambda
>```

---

### 2. Stage
The Stage function will:
- take as inputs:
  - a tensor (**`x`**)
  - the number of channels (also called filters) (**`channels`**)
  - the number of repetitions of the second part of the stage (**`repetitions`**)
  - the number of groups for the Group Convolution blocks (**`groups`**)
- run:
  - apply a Shufflenet block with strides=2
  - apply **`repetitions`** times a Shufflenet block with strides=1
- return the tensor

**Code:**
>```python
>def stage(x, channels, repetitions, groups):
>     x = shufflenet_block(x, channels=channels, strides=2, groups=groups)
>     for i in range(repetitions):
>         x = shufflenet_block(x, channels=channels, strides=1, groups=groups)
>     return x
>```

---

### 3. Shufflenet block
The Shufflenet block will:
- take as inputs:
  - a tensor (**`tensor`**)
  - the number of channels (**`channels`**)
  - the strides (**`strides`**)
  - the number of groups for the Group Convolution blocks (**`groups`**)
- run:
  - apply a Group Convolution block with 1/4 **`channels`** channels followed by *Batch Normalizaion-ReLU*
  - apply **`Channel Shuffle`** to this tensor
  - apply a *Depthwise Convolution* layer followed by *Batch Normalizaion*
  - if **`strides`** is 2:
    - subtract from **`channels`** the number of channels of **`tensor`** so that after the concatenation the output tensor will have **`channels`** channels
  - apply a Group Convolution block with **`channels`** channels  followed by *Batch Normalizaion*
  - if **`strides`** is 1:
    - *add* this tensor with the input **`tensor`**
  - else:
    - apply a 3x3 *Average Pool* with strides 2 (**[vi]**) to the input **`tensor`** and *concatenate* it with this tensor
  - apply *ReLU* activation to the tensor
- return the tensor

Note that according to **[iv]** we should not apply Group Convolution to the first inupt (24 channels) and apply only the Convolution operation instead which we can code with a simple `if-else` statement. However, for the sake of clarity of the code we ommit it.

**Code:**
>```python
>def shufflenet_block(tensor, channels, strides, groups):
>     x = gconv(tensor, channels=channels // 4, groups=groups)
>     x = BatchNormalization()(x)
>     x = ReLU()(x)
>
>     x = channel_shuffle(x, groups)
>     x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same')(x)
>     x = BatchNormalization()(x)
>
>     if strides == 2:
>         channels = channels - tensor.get_shape().as_list()[-1]
>     x = gconv(x, channels=channels, groups=groups)
>     x = BatchNormalization()(x)
>
>     if strides == 1:
>         x = Add()([tensor, x])
>     else:
>         avg = AvgPool2D(pool_size=3, strides=2, padding='same')(tensor)
>         x = Concatenate()([avg, x])
>
>     output = ReLU()(x)
>     return output
>```

---

### 4. Group Convolution
The Group Convolution function will:
- take as inputs:
  - a tensor (**`tensor`**)
  - the number of channels of the output tensor (**`channels`**)
  - the number of groups (**`groups`**)
- run:
  - get the number of channels (**`input_ch`**) of the input tensor using the get_shape() method
  - calculate the number of channels per group (**`group_ch`**) by dividing **`input_ch`** by **`groups`**
  - calculate how many channels will have each group after the Convolution layer. It will be equal to **`channels`** divided by **`groups`**
  - for every group:
    - get the **`group_tensor`** which will be a sub-tensor of **`tensor`** with specific channels
    - apply a 1x1 Convolution layer with **`output_ch`** channels
    - add the tensor to a list (**`groups_list`**)
  - *Concatenate* all the tensors of **`groups_list`** to one tensor
- return the tensor

Note that there is a commented line in the code bellow. One can get a slice of a tensor by using the simple slicing notation `a[:, b:c, d:e]` but the code takes too long to run (as it is in the case of tensorflow.slice()). By using a Lambda layer and applying it on the tensor we have the same result but much faster.

**Code:**
>```python
>def gconv(tensor, channels, groups):
>     input_ch = tensor.get_shape().as_list()[-1]
>     group_ch = input_ch // groups
>     output_ch = channels // groups
>     groups_list = []
>
>     for i in range(groups):
>         group_tensor = tensor[:, :, :, i * group_ch: (i+1) * group_ch]
>         # group_tensor = Lambda(lambda x: x[:, :, :, i * group_ch: (i+1) * group_ch])(tensor)
>         group_tensor = Conv2D(output_ch, 1)(group_tensor)
>         groups_list.append(group_tensor)
>
>     output = Concatenate()(groups_list)
>     return output
>```

---

### 5. Channel Shuffle
The Channel Shuffle function will:
- take as inputs:
  - a tensor (**`x`**)
  - the number of groups (**`groups`**)
- run:
  - get the dimensions (**`width, height, channels`**) of the input tensor. Note that the first number of `x.get_shape().as_list()` will be the batch size.
  - calculate the number of channels per group (**`group_ch`**)
  - reshape **`x`** to **`width`** x **`height`** x **`group_ch`** x **`groups`**
  - permute the last two dimensions of the tensor (**`group_ch`** x **`groups`** -> **`groups`** x **`group_ch`**)
  - reshape **`x`** to its original shape (**`width`** x **`height`** x **`channels`**)
- return the tensor

**Code:**
>```python
>def channel_shuffle(x, groups):  
>     _, width, height, channels = x.get_shape().as_list()
>     group_ch = channels // groups
>
>     x = Reshape([width, height, group_ch, groups])(x)
>     x = Permute([1, 2, 4, 3])(x)
>     x = Reshape([width, height, channels])(x)
>     return x
>```

---

### 6. Stem of the model
Now we can start coding the model. We will start with the model's stem. According to **[v]** the first layer of the model is a 3x3 Convolution layer with 24 filters followed by (**[iii]**) a BatchNormalization and a ReLU activation.

The next layer is a 3x3 Max Pool with strides 2.

**Code:**
>```python
>input = Input([224, 224, 3])
>x = Conv2D(filters=24, kernel_size=3, strides=2, padding='same')(input)
>x = BatchNormalization()(x)
>x = ReLU()(x)
>x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
>```

---

### 7. Main part of the model
The main part of the model consists of **`Stage`** blocks. We first define the hyperparameters **`repetitions`**, **`initial_channels`** acoording to **[v]** and **`groups`**. Then for each number of repetitions we calculate the number of channels according to **[i]** and apply the `stage()` function on the tensor.

**Code:**
>```python
>repetitions = 3, 7, 3
>initial_channels = 384
>groups = 8
>
>for i, reps in enumerate(repetitions):
>     channels = initial_channels * (2**i)
>     x = stage(x, channels, reps, groups)
>```

---

### 8. Rest of the model
The model closes with a Global Pool layer and a Fully Connected one with 1000 classes (**[v]**).

**Code:**
>```python
>x = GlobalAvgPool2D()(x)
>output = Dense(1000, activation='softmax')(x)
>
>from tensorflow.keras import Model
>model = Model(input, output)
>```

---

## Final code

**Code:**
```python
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, \
     Dense, Concatenate, Add, ReLU, BatchNormalization, AvgPool2D, \
     MaxPool2D, GlobalAvgPool2D, Reshape, Permute, Lambda


def stage(x, channels, repetitions, groups):
    x = shufflenet_block(x, channels=channels, strides=2, groups=groups)
    for i in range(repetitions):
        x = shufflenet_block(x, channels=channels, strides=1, groups=groups)
    return x


def shufflenet_block(tensor, channels, strides, groups):
    x = gconv(tensor, channels=channels // 4, groups=groups)
    x = BatchNormalization()(x)
    x = ReLU()(x)
 
    x = channel_shuffle(x, groups)
    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
 
    if strides == 2:
        channels = channels - tensor.get_shape().as_list()[-1]
    x = gconv(x, channels=channels, groups=groups)
    x = BatchNormalization()(x)
 
    if strides == 1:
        x = Add()([tensor, x])
    else:
        avg = AvgPool2D(pool_size=3, strides=2, padding='same')(tensor)
        x = Concatenate()([avg, x])
 
    output = ReLU()(x)
    return output


def gconv(tensor, channels, groups):
    input_ch = tensor.get_shape().as_list()[-1]
    group_ch = input_ch // groups
    output_ch = channels // groups
    groups_list = []
 
    for i in range(groups):
        # group_tensor = tensor[:, :, :, i * group_ch: (i+1) * group_ch]
        group_tensor = Lambda(lambda x: x[:, :, :, i * group_ch: (i+1) * group_ch])(tensor)
        group_tensor = Conv2D(output_ch, 1)(group_tensor)
        groups_list.append(group_tensor)
 
    output = Concatenate()(groups_list)
    return output


def channel_shuffle(x, groups):  
    _, width, height, channels = x.get_shape().as_list()
    group_ch = channels // groups
 
    x = Reshape([width, height, group_ch, groups])(x)
    x = Permute([1, 2, 4, 3])(x)
    x = Reshape([width, height, channels])(x)
    return x


input = Input([224, 224, 3])
x = Conv2D(filters=24, kernel_size=3, strides=2, padding='same')(input)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)


repetitions = 3, 7, 3
initial_channels = 384
groups = 8
 
for i, reps in enumerate(repetitions):
    channels = initial_channels * (2**i)
    x = stage(x, channels, reps, groups)


x = GlobalAvgPool2D()(x)
output = Dense(1000, activation='softmax')(x)
 
from tensorflow.keras import Model
model = Model(input, output)
```

---

## Model diagram

<img src="https://raw.githubusercontent.com/Machine-Learning-Tokyo/CNN-Architectures/master/Implementations/ShuffleNet/ShuffleNet_diagram.svg?sanitize=true">
