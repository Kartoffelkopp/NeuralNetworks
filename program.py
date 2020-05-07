import neuralnetwork as nn
import numpy as np

with np.load("mnist.npz") as data:
    training_images = data["training_images"]
    training_labels = data["training_labels"]

print(training_labels[0])

layer_sizes = (3,5,10)
x = np.ones((layer_sizes[0],1))

net = nn.NeuralNetwork(layer_sizes)
prediction = net.predict(x)

print(prediction)