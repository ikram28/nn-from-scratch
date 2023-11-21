import numpy as np

class sigmoid_activation:
    def __init__(self):
        self.name = "sigmoid"
        self.Y = None

    def feed_forward(self,in_matrix):
        self.Y = 1 / (1 + np.exp(-in_matrix))
        return self.Y

    def backpropagation(self, dL_dY):
        return np.multiply(dL_dY, np.multiply(self.Y, 1-self.Y))





