# CNN Model

## Overview
The CNN model implemented in this project is a simple Convolutional Neural Network designed for image classification tasks, specifically mimicking MNIST-like data processing.

## Architecture
- **Input**: 28x28 grayscale images (1 channel).
- **Layers**:
    - Conv2d (1 -> 32 filters, 3x3 kernel) + ReLU + MaxPool
    - Conv2d (32 -> 64 filters, 3x3 kernel) + ReLU + MaxPool
    - Flatten
    - Linear (64*7*7 -> 128) + ReLU
    - Linear (128 -> 10) (Output logits)

## Drift Detection Features
For drift detection, we extract embeddings from the penultimate layer (the 128-dimensional output of the first fully connected layer). This allows us to monitor changes in the learned feature space.
