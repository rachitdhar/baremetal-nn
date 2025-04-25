# baremetal-nn
A ground-up implementation of a neural network in C, from scratch, and without using any AI/ML libraries.

## Features

- Can create a Neural Network (NN) with an input layer, hidden layers, and output layer, all with a custom choice of the number of nodes.
- Can add the activation function to be used for each layer (currently, ReLU function has been implemented, but has been designed so as to easily create more activation functions).
- Can choose the loss/cost function to use (currently, MSE loss is implemented, but has been designed so as to easily create different loss functions, and their differentiated functions).
- Performs linear regression to calculate node values, and gradient descent during backpropagation to adjust weights and biases.
- Can pass input and target datasets through CSV files - for both training and testing.
- Optionally allows the ability to print the details of the neural network during training (i.e., node values, weights and biases for the layers), which may help during debugging.
- Returns the prediction set and accuracy vector after the test is complete.

## Unique Design Choices

- Implemented in C, instead of a typical language like Python, to consciously focus on the logic and design of the neural network rather than relying on language tricks and features that make things easy but obscure. Additionally, C is fundamentally faster in execution, which makes this implementation both clear and powerful to use. 
- Not using any AI/ML libraries in C. Building functions required from scratch. This is done to keep things open and simple, and removing unnecessary "black boxes" introduced as a result of using AI/ML functions whose working may be unknown.
