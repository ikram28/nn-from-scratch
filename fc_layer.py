import numpy as np

class fc_layer:
    def __init__(self, in_size, out_size):
        self.weights = np.zeros((out_size, in_size+1))
        self.name = "FC"
        self.dL_dW = None
        self.Z = None


    def init_params(self):
        self.weights = np.random.normal(0, 1, self.weights.shape)
        return self.weights

    def feed_forward(self, in_matrix):
        self.Z = np.concatenate((np.ones((1,in_matrix.shape[1])),in_matrix),axis=0)
        Y = np.dot(self.weights,self.Z)
        return Y

# shape of in_matrix = (num_features, batch_size)

    def backpropagation(self, dL_dY):
        self.dL_dW = np.dot(dL_dY, self.Z.T)
        dL_dZ =  np.dot(self.weights.T, dL_dY)
        return dL_dZ[1:]




