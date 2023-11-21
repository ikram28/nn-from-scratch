import cv2
import numpy as np
from neural_net import neural_net

image = cv2.imread("9.jpg",0)/255
image = cv2.resize(image, (100,100))
flatten_image = image.flatten()


layer_dims = [len(flatten_image), 1 ,len(flatten_image)]

# Initialize the neural network
net = neural_net(layer_dims)

X = flatten_image.reshape(-1, 1)
Y = np.flip(flatten_image.reshape(-1, 1))


print(X.shape)

# Initialize network parameters
net.create_layers()
net.init_params()

# Step 1: Feedforward
output = net.feed_forward(X)

#show the output image
output_image = output.reshape(image.shape)



# Step 3 : Training
prediction = net.descent_gradient(X,Y,0.001, 10000)

#show the predicted image after training
predicted_image = prediction.reshape(image.shape)


cv2.imshow("9", image)
cv2.imshow("Output Image", output_image)
cv2.imshow("Output Image After Training", predicted_image)


cv2.waitKey(0)
cv2.destroyAllWindows()







