# SimpleFNN

Simple ANN in C++

A minimal implementation of a feed-forward neural network (FNN) from scratch in C++, utilizing the Eigen library for matrix operations. This project is intended as an educational tool to demonstrate the basics of neural network architecture, including forward propagation, backpropagation, training, and evaluation.

Features

    Feed-forward Propagation: Processes inputs through sequential layers to predict outputs.
    Backpropagation: Adjusts the network's weights and biases based on the error between predicted and actual outputs.
    Sigmoid Activation Function: Used for non-linear activation of neurons.
    Binary Classification: Supports binary classification problems with a single output neuron.
    Epoch-based Training: Allows training the model for a specified number of epochs with a given learning rate.
    Accuracy Evaluation: Evaluates the model's performance on a given dataset by calculating accuracy.

Dependencies

    C++ Compiler: GCC, Clang, or any C++11 compatible compiler.
    Eigen Library: A high-level C++ library for linear algebra, matrix and vector operations, numerical solvers, and related algorithms.

Setup

    Install Eigen: This project depends on the Eigen library. You can download and install Eigen from Eigen's official website. Ensure that Eigen is properly included in your project's include path.

    Compilation: Compile the project using a C++ compiler that supports C++11 or later. For example, using g++:

    sh

    g++ -I /path/to/eigen -o simple_ann SimpleAnn.cpp

    Replace /path/to/eigen with the path to your Eigen installation.

Usage

To run the Simple ANN model, execute the compiled binary:

sh

./simple_ann

This will train the neural network on a predefined dataset (included in main()) and evaluate its accuracy.

Customizing the Model

You can customize the neural network by modifying the layerSizes parameter in the SimpleAnn constructor within main(). Each element in the layerSizes vector represents the number of neurons in each layer, including the input and output layers.
How It Works

    Initialization: The constructor initializes the network with random weights and biases based on the provided layer sizes.
    Training: During training, the network uses forward propagation to predict outputs and backpropagation to adjust weights and biases.
    Activation Function: The sigmoid function is used for activation, with its derivative supporting the backpropagation process.
    Evaluation: Post-training, the network's performance can be evaluated on a dataset to calculate accuracy.

Note

This project is for educational purposes and demonstrates basic concepts of neural networks. For production-level projects, consider using a dedicated deep learning framework.
