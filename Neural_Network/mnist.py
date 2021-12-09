import numpy as np

from fc_layer import FCLayer
from activation_layer import ActivationLayer
from network import Network
from loss import mse, mse_prime
from activations import tanh, tanh_prime, sigmoid, sigmoid_prime

from keras.datasets import mnist
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, 28*28)
X_train = X_train.astype('float32')
X_train = X_train / 255
y_train = np_utils.to_categorical(y_train)

X_test = X_test.reshape(X_test.shape[0], 1, 28*28)
X_test = X_test.astype('float32')
X_test = X_test / 255
y_test = np_utils.to_categorical(y_test)

net = Network()
net.add(FCLayer(28*28, 100))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(100, 50))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(50, 10))
net.add(ActivationLayer(tanh, tanh_prime))

net.use(mse, mse_prime)
net.fit(X_train[:1000], y_train[:1000], epochs=35, learning_rate=0.1)

out = net.predict(X_test[:3])
print(out)
y_predicted = [np.argmax(i) for i in out]
print(y_predicted)
print(y_test[:3])
