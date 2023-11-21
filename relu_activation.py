import numpy as np

class relu_activation:
    def __init__(self):
        self.name = "relu"
        self.Z = None

    def feed_forward(self, in_matrix):
        self.Z = in_matrix
        return np.maximum(0, self.Z)

    def backpropagation(self,dL_dY):
        return dL_dY * (self.Z > 0)
