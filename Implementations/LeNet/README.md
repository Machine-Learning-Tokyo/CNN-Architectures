# LeNet-5: Gradient-Based Learning Applied to Document Recognition

LeNet-5 is a highly efficient convolutional neural network (CNN) designed for handwritten character recognition. This model was introduced in the paper [*Gradient-Based Learning Applied to Document Recognition*](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) by **Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner**. The paper was published in the *Proceedings of the IEEE (1998)*.

![LeNet-5 Architecture](https://raw.githubusercontent.com/entbappy/Branching-tutorial/master/lenet/lenet-5.png)

## Table of Contents
- [Introduction](#introduction)
- [Network Structure](#network-structure)
  - [Input Layer](#input-layer)
  - [C1: Convolutional Layer](#c1-convolutional-layer)
  - [S2: Pooling Layer (Downsampling)](#s2-pooling-layer-downsampling)
  - [C3: Convolutional Layer](#c3-convolutional-layer)
  - [S4: Pooling Layer (Downsampling)](#s4-pooling-layer-downsampling)
  - [C5: Convolutional Layer](#c5-convolutional-layer)
  - [F6: Fully Connected Layer](#f6-fully-connected-layer)
  - [Output Layer: Fully Connected Layer](#output-layer-fully-connected-layer)
- [Summary](#summary)
- [References](#references)

## Introduction

LeNet-5 is a seminal convolutional neural network architecture that laid the foundation for modern deep learning techniques. It was primarily designed for recognizing handwritten digits and characters, a task that was challenging before the advent of deep learning.

The network consists of several convolutional layers followed by pooling (downsampling) layers and fully connected layers, making it highly efficient in extracting features and reducing the dimensionality of input data.

## Network Structure

LeNet-5 is composed of seven layers (excluding the input layer), each containing trainable parameters. It incorporates the basic modules of deep learning: convolutional layers, pooling layers, and fully connected layers. The architecture of LeNet-5 is foundational for many other deep learning models.

### Input Layer

- **Image Size**: 32x32 (normalized input image)
- *Note*: The input layer is not counted as part of the network structure.

### C1: Convolutional Layer

- **Input Size**: 32x32
- **Convolution Kernel**: 5x5
- **Number of Filters**: 6
- **Output Size**: 28x28
- **Number of Neurons**: 28x28x6
- **Trainable Parameters**: 156 (6 filters with 25 weights and 1 bias each)
- **Connections**: 122,304

**Description**:
- The first convolution operation is performed on the input image using 6 convolution kernels of size 5x5, resulting in 6 feature maps of size 28x28.

### S2: Pooling Layer (Downsampling)

- **Input Size**: 28x28
- **Pooling Area**: 2x2
- **Output Size**: 14x14
- **Number of Feature Maps**: 6
- **Trainable Parameters**: 12
- **Connections**: 5,880

**Description**:
- Pooling is performed using 2x2 kernels, producing 6 feature maps of size 14x14. Each feature map in S2 is a downsampled version of the corresponding feature map in C1.

### C3: Convolutional Layer

- **Input Size**: 14x14
- **Convolution Kernel**: 5x5
- **Number of Filters**: 16
- **Output Size**: 10x10
- **Trainable Parameters**: 1,516
- **Connections**: 151,600

**Description**:
- This layer consists of 16 convolutional filters, each producing a 10x10 feature map. The feature maps in C3 are a combination of the feature maps in S2.

### S4: Pooling Layer (Downsampling)

- **Input Size**: 10x10
- **Pooling Area**: 2x2
- **Output Size**: 5x5
- **Number of Feature Maps**: 16
- **Trainable Parameters**: 32
- **Connections**: 2,000

**Description**:
- Similar to S2, this layer performs downsampling, producing 16 feature maps of size 5x5. Each feature map in S4 is a downsampled version of the corresponding feature map in C3.

### C5: Convolutional Layer

- **Input Size**: 5x5
- **Convolution Kernel**: 5x5
- **Number of Filters**: 120
- **Output Size**: 1x1
- **Trainable Parameters**: 48,120
- **Connections**: 48,120

**Description**:
- The C5 layer performs convolution with 120 filters, each of size 5x5, producing a 1x1 output. This layer is fully connected to the previous layer's feature maps.

### F6: Fully Connected Layer

- **Input**: 120-dimensional vector
- **Output**: 84-dimensional vector
- **Trainable Parameters**: 10,164

**Description**:
- The F6 layer is a fully connected layer with 84 nodes. Each node corresponds to a possible output class and is connected to all nodes in the previous layer.

### Output Layer: Fully Connected Layer

- **Number of Nodes**: 10 (corresponding to digits 0-9)
- **Trainable Parameters**: 840
- **Connections**: 840

**Description**:
- The output layer consists of 10 nodes, each representing a digit from 0 to 9. The final output is determined by the node with the highest activation.

## Summary

LeNet-5 demonstrates the power of convolutional neural networks in efficiently recognizing handwritten characters. Its use of local connections, shared weights, and pooling layers significantly reduces the number of parameters, making it an efficient model for document recognition.

## References

- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). [*Gradient-Based Learning Applied to Document Recognition*](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf). *Proceedings of the IEEE*.
