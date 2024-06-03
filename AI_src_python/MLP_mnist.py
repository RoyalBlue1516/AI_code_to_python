import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import math
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, stratify=y_train, test_size=0.2)


x_train = x_train.reshape(x_train.shape[0], 28*28)
x_dev = x_dev.reshape(x_dev.shape[0], 28*28)
x_test = x_test.reshape(x_test.shape[0], 28*28)


x_train = x_train.astype('float32') / 255.
x_dev = x_dev.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# one-hot encoding
y_train = keras.utils.to_categorical(y_train)
y_dev = keras.utils.to_categorical(y_dev)
y_test = keras.utils.to_categorical(y_test)

class MultiLayer:
    def __init__(self, learning_rate=0.01, batch_size=100, hidden_layers=[100]):
        self.weights = []  
        self.biases = []  
        self.lr = learning_rate
        self.batch_size = batch_size
        self.hidden_layers = hidden_layers
        self.best_loss = float("inf")
        self.early_stopping = 0

    def init_weights(self, input_size, output_size):
        layer_sizes = [input_size] + self.hidden_layers + [output_size]
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.normal(0, 1, (layer_sizes[i], layer_sizes[i+1])))
            self.biases.append(np.zeros(layer_sizes[i+1]))

    def sigmoid(self, z):
        z = np.clip(z, -100, None)
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        z = np.clip(z, -100, None)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, x):
        activations = [x]
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(activations[-1], w) + b
            a = self.sigmoid(z)
            activations.append(a)
        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        a = self.softmax(z)
        activations.append(a)
        return activations

    def loss(self, x, y):
        a = self.forward(x)[-1]
        a = np.clip(a, 1e-10, 1-1e-10)
        return -np.sum(y * np.log(a))

    def backpropagation(self, x, y):
        activations = self.forward(x)
        deltas = [activations[-1] - y]
        for l in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(deltas[-1], self.weights[l].T) * activations[l] * (1 - activations[l])
            deltas.append(delta)
        deltas.reverse()

        grad_w = []
        grad_b = []
        for i in range(len(self.weights)):
            grad_w.append(np.dot(activations[i].T, deltas[i]))
            grad_b.append(np.sum(deltas[i], axis=0))

        return grad_w, grad_b

    def minibatch(self, x, y):
        iter = math.ceil(len(x) / self.batch_size)
        x, y = shuffle(x, y)
        for i in range(iter):
            start = self.batch_size * i
            end = self.batch_size * (i + 1)
            yield x[start:end], y[start:end]

    def fit(self, x_data, y_data, epochs=100, x_dev=None, y_dev=None, x_test=None, y_test=None):
        self.init_weights(x_data.shape[1], y_data.shape[1])
        for epoch in range(epochs):
            l = 0
            for x, y in self.minibatch(x_data, y_data):
                l += self.loss(x, y)
                grad_w, grad_b = self.backpropagation(x, y)
                for i in range(len(self.weights)):
                    self.weights[i] -= self.lr * grad_w[i] / len(x)
                    self.biases[i] -= self.lr * grad_b[i] / len(x)

            val_loss = self.val_loss(x_dev, y_dev)
            if val_loss < self.best_loss:
                self.best_loss = val_loss
            else:
                self.early_stopping += 1

            if self.early_stopping == 5:
                print(f'best_val_loss: {self.best_loss:.4f}')
                print('early-stop')
                break


            print(f'epoch({epoch+1}) ===> loss : {l/len(y_data):.5f} | val_loss : {val_loss:.5f}', \
                  f' | train_accuracy : {self.score(x_data, y_data):.5f} | dev_accuracy : {self.score(x_dev, y_dev):.5f} | test_accuracy : {self.score(x_test, y_test):.5f}')

    def predict(self, x_data):
        a = self.forward(x_data)[-1]
        return np.argmax(a, axis=1)

    def score(self, x_data, y_data):
        return np.mean(self.predict(x_data) == np.argmax(y_data, axis=1))

    def val_loss(self, x_dev, y_dev):
        val_loss = self.loss(x_dev, y_dev)
        return val_loss / len(y_dev)

model = MultiLayer(learning_rate=0.1, batch_size=256, hidden_layers=[100,50,30])
model.fit(x_train, y_train, epochs=10, x_dev=x_dev, y_dev=y_dev, x_test=x_test, y_test=y_test)
