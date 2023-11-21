from fc_layer import fc_layer
from relu_activation import relu_activation
import numpy as np

class neural_net:
    def __init__(self, layer_dim):
        self.layer_dim = layer_dim
        self.layers = []


    def create_layers(self):
        for i in range(len(self.layer_dim) - 1):
            self.layers.append(fc_layer(self.layer_dim[i], self.layer_dim[i+1]))
            self.layers.append(relu_activation())

    def init_params(self):
        for layer in self.layers:
            if layer.name == "FC":
                layer.init_params()

    def feed_forward(self, input_data):
        output = input_data

        for layer in self.layers:
            output = layer.feed_forward(output)

        return output

    def backpropagation(self, dL_dY):
        dL_dZ = dL_dY
        for layer in reversed(self.layers):
            dL_dZ = layer.backpropagation(dL_dZ)
        return dL_dZ

    def descent_gradient(self,X, y, learning_rate , num_iterations):
        for iteration in range(num_iterations):
            predictions = self.feed_forward(X)
            loss = np.power((predictions - y),2)
            self.backpropagation(loss)

            for layer in self.layers:
                if layer.name == "FC":
                    layer.weights -= learning_rate * layer.dL_dW

            print(f"Iteration {iteration}, Loss: {np.sum(loss)}")

        return predictions