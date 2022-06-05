import numpy as np
import matplotlib.pyplot as plt

inputs = np.array([[0, 0, 0],
                   [0, 0, 1],
                   [0, 1, 0],
                   [0, 1, 1],
                   [1, 0, 0],
                   [1, 0, 1],
                   [1, 1, 1]])

outputs = np.array([[0], [0], [0], [0], [0], [1], [1]])




class NeuralNetwork:

    def __init__(self, inputs, outputs):
        self.inputs  = inputs
        self.outputs = outputs
        self.weights = np.array([[.50], [.50], [.50]])
        self.error_history = []
        self.epoch_list = []


    def sig(self, x, deriv=False):
        if deriv == True:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def feed_forward(self):
        self.hidden = self.sig(np.dot(self.inputs, self.weights))

    def backpropagation(self):
        self.error  = self.outputs - self.hidden
        delta = self.error * self.sig(self.hidden, deriv=True)
        self.weights += np.dot(self.inputs.T, delta)


    def train(self, epochs=10000):
        for epoch in range(epochs):
            self.feed_forward()

            self.backpropagation()

            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch)


    def result(self, new_input):
        result = self.sig(np.dot(new_input, self.weights))
        return result





NN = NeuralNetwork(inputs, outputs)

NN.train()


example = np.array([[1, 1, 1]])
example_2 = np.array([[0, 1, 1]])


print(NN.result(example), ' - Ожидаемый ответ: 1')
print(NN.result(example_2), ' - Ожидаемый ответ: 0')


plt.figure(figsize=(10,5))
plt.plot(NN.epoch_list, NN.error_history)
plt.xlabel('Количество эпох')
plt.ylabel('Ошибки')
plt.show()