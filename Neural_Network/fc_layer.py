from layer import Layer
import numpy as np

class FCLayer(Layer):

    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size)-0.5
        self.bias = np.random.rand(1, output_size)-0.5

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        # input_error = dE/dY * transpose(W)
        input_error = np.dot(output_error, self.weights.T)

        # weights_error = transpose(X) * dE/dY
        weights_error = np.dot(self.input.T, output_error)

        self.weights = self.weights - learning_rate * weights_error
        self.bias = self.bias - learning_rate * output_error

        return input_error