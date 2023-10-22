# PyTorch Tutorial: Fashion MNIST with Convolutional Neural Networks (CNNs)

Welcome to a comprehensive tutorial on the Fashion MNIST dataset using PyTorch. Fashion MNIST is a dataset of 60,000 28x28 grayscale images of 10 fashion categories, along with a test set of 10,000 images.

## Overview

The Fashion MNIST dataset is a collection of grayscale images of 10 fashion categories, each of size 28x28 pixels. It's used as a drop-in replacement for the classic MNIST dataset. It serves as a more challenging classification problem than the regular MNIST digit dataset due to the similarities in clothing items.

![Fashion MNIST Sample](https://github.com/zalandoresearch/fashion-mnist/raw/master/doc/img/fashion-mnist-sprite.png)

Each image in the dataset corresponds to a label from 0-9, representing the ten categories:

| Label | Description |
| --- | --- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## Table of Contents:

1. [PyTorch Tutorial: Fashion MNIST with Convolutional Neural Networks (CNNs)](#pytorch-tutorial-fashion-mnist-with-convolutional-neural-networks-cnns)
2. [Prerequisites](#prerequisites)
3. [Setup and Installation](#setup-and-installation)
4. [Understanding the Fashion MNIST Dataset](#understanding-the-fashion-mnist-dataset)
5. [Data Loading and Preprocessing](#data-loading-and-preprocessing)
6. [Understanding PyTorch Utilities](#understanding-pytorch-utilities)
   - [torchvision](#torchvision)
   - [torch.utils](#torchutils)
   - [DataLoader](#dataloader)
7. [Convolutional Neural Networks (CNNs)](#convolutional-neural-networks-cnns)
8. [Training and Evaluation](#training-and-evaluation)
   - [Model Instantiation](#model-instantiation)
   - [Optimizer and Loss Function](#optimizer-and-loss-function)
9. [Understanding Optimizers and Loss Functions](#understanding-optimizers-and-loss-functions)
10. [Training Loop (Explained)](#training-loop-explained)
11. [Evaluation](#evaluation)
12. [Advanced Topics: Regularization with Dropout](#advanced-topics-regularization-with-dropout)
13. [Understanding the Role of `F.log_softmax`](#understanding-the-role-of-flog_softmax)
14. [Training](#training)
15. [Logging and Observing the Loss During Training](#logging-and-observing-the-loss-during-training)
16. [Reusing the Evaluation Code and The Importance of Testing on a Validation Set](#reusing-the-evaluation-code-and-the-importance-of-testing-on-a-validation-set)
17. [Advanced Topics: Data Augmentation](#advanced-topics-data-augmentation)
18. [Conclusion](#conclusion)
19. [References and Acknowledgments](#references-and-acknowledgments)
20. [Exercises for Practice](#exercises-for-practice)
21. [Additional Resources](#additional-resources)
22. [Feedback](#feedback)
23. [Exercise Solutions](#exercise-solutions)

## Prerequisites

Before diving into the tutorial, ensure you have the following prerequisites installed and set up:

1. **Python**: This tutorial requires Python 3.x. Python is the primary language we'll be using.
2. **PyTorch & torchvision**: PyTorch is an open-source machine learning library, and torchvision offers datasets and models for computer vision.
3. **Jupyter Notebook**: The interactive environment where this tutorial is presented.
4. **NumPy**: A library for numerical operations in Python.
5. **scikit-learn**: Machine learning library in Python. We'll use it for performance metrics.
6. **Seaborn & Matplotlib**: Visualization libraries in Python.
7. **CUDA (Optional)**: If you have a compatible NVIDIA GPU, you can install CUDA for GPU acceleration with PyTorch.

## About this Tutorial

This tutorial was meticulously crafted by **Muhammad Junaid Ali Asif Raja** for a seminar/workshop held on 27th October 2023. The main aim is to provide a comprehensive understanding of implementing CNNs using PyTorch, targeting both beginners and intermediate learners.

## Getting Started

Follow these steps to get started with the tutorial:

### 1. Clone the Repository:

```bash
git clone https://github.com/junaidaliop/pytorch-fashionMNIST-tutorial.git
cd pytorch-fashionMNIST-tutorial
```


### 2. Set Up the Conda Environment:

Use the provided `PyTorchTutorial.xml` file to set up the Conda environment with all required dependencies:

```bash
conda env create -f PyTorchTutorial.xml
```

Activate the environment:

```bash
conda activate PyTorchTutorial
```

### 3. Launch the Jupyter Notebook:

```bash
jupyter notebook
```

Navigate to the `pytorch_fashion_mnist_tutorial.ipynb` file in the Jupyter Notebook interface, and you're ready to dive into the tutorial!

## Acknowledgements

Special thanks to **Dr. Naveed Ishtiaq Chaudhary** for presenting me with the opportunity.
