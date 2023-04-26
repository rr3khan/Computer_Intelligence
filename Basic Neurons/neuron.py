# ECE 457B Computer intelligence Assignment 1 Riyad Khan
# Q1 B

import numpy as np
import matplotlib.pyplot as plt
import random

# helper class for the 3D Point Data
class Three_d_data:
    def __init__(
        self, target_c: int, bias: int, three_d_coordinate: np.ndarray
    ) -> None:
        self.target_c = target_c
        self.bias = bias
        self.three_d_coordinates = three_d_coordinate

    def log(self) -> None:
        print("Class:", self.target_c)
        print("Bias:", self.bias)
        print("Coordinate:", self.three_d_coordinates)

class Neuron:
    def __init__(
        self,
        input_data: list,
        neuron_type: str,
        learning_rate: float,
        stopping_threshold: float,
        max_epochs=10000,
    ):
        # parse input data
        self.input_coordinates = []
        self.targets = []
        self.input_bias = []
        self.delta_weights = np.array([])
        self.delta_threshold = stopping_threshold
        self.max_epochs = max_epochs
        self.elapsed_epochs = 0

        self.input_data = input_data

        for input in input_data:
            self.input_coordinates.append(input.three_d_coordinates)
            self.targets.append(input.target_c)
            self.input_bias.append(input.bias)

        # weights randomly selected in interval [-1, 1]
        # The number of weights is based on the dimension of our data hence since we are working with 3D
        # data we will have 3 weights
        self.weights = np.random.uniform(-1, 1, (len(self.input_coordinates[0])))
        # type adaline or perceptron
        self.neuron_type = neuron_type
        # neta learning rate
        self.learning_rate = learning_rate

    def update_weights(self, sample_data, sample_index) -> np.ndarray:
        self.delta_weights = np.array([])

        # The number of weights is the same as the dimension of the inputs
        # The number of targets is the same as the number of inputs
        # use the dot product to get the sum of the weights multiplied by the inputs
        sum_prod = np.dot(sample_data, self.weights)

        if self.neuron_type == "adaline":
            # for the adaline update using the Least mean square algorithm
            # also called the Windrow-Hoff learning rule

            # online training hence update weights after seeing each data point
            for w_index, weight in enumerate(self.weights):
                sig = sigmoid(sum_prod)

                error = self.targets[sample_index] - sig
                # print("error", error)
                # LMS_update = self.learning_rate * error*(sig*sig*np.exp(-sum_prod))*sample_data[w_index]
                # DO NOT USE exp it blows up!
                LMS_update = (
                    self.learning_rate
                    * (error)
                    * (sig * (1 - sig))
                    * sample_data[w_index]
                )
                self.delta_weights = np.append(self.delta_weights, LMS_update)

        elif self.neuron_type == "perceptron":
            output_o = step(sum_prod)

            for w_index, weight in enumerate(self.weights):
                error = self.targets[sample_index] - output_o
                HLR_update = self.learning_rate * (error) * sample_data[w_index]
                self.delta_weights = np.append(self.delta_weights, HLR_update)

        self.weights = np.add(self.weights, self.delta_weights)
        return self.delta_weights

    def train(self):
        self.elapsed_epochs = 0

        # while self.elapsed_epochs < self.num_epochs:
        # While the delta weights have not converged close to zero keep learning
        # or stop and break if we take too long and exceed max epochs
        while all(abs(dw) > self.delta_threshold for dw in self.delta_weights):
            for sample_index, data_point in enumerate(self.input_coordinates):
                # print("Data", data_point)
                self.update_weights(data_point, sample_index)

            self.elapsed_epochs += 1
            if self.elapsed_epochs == self.max_epochs:
                break

    def classify_point(self, point):
        raw_class = np.dot(self.weights, point)
        binary_class = step(raw_class)
        return raw_class, binary_class

    def log(self) -> None:
        print("Values returned by:", self.neuron_type)
        print("Weights", self.weights)
        # print("Coordinates", self.input_coordinates)
        # print("Targets", self.targets)
        print("Delta weights", self.delta_weights)
        print("Num epochs", self.elapsed_epochs)

# helper functions

# sigmoid function
def sigmoid(x: np.ndarray) -> float:
    return 1 / (1 + np.exp(-x))


# step function
def step(input):
    return 1 if input > 0 else 0

x1 = Three_d_data(0, -1, np.array([-1, 0.8, 0.7, 1.2]))
x2 = Three_d_data(0, -1, np.array([-1, -0.8, -0.7, 0.2]))
x3 = Three_d_data(0, -1, np.array([-1, -0.5, 0.3, -0.2]))
x4 = Three_d_data(0, -1, np.array([-1, -2.8, -0.1, -2]))

y1 = Three_d_data(1, -1, np.array([-1, 1.2, -1.7, 2.2]))
y2 = Three_d_data(1, -1, np.array([-1, -0.8, -2, 0.5]))
y3 = Three_d_data(1, -1, np.array([-1, -0.5, -2.7, -1.2]))
y4 = Three_d_data(1, -1, np.array([-1, 2.8, -1.4, 2.1]))

adaline = Neuron([x1, x2, x3, x4, y1, y2, y3, y4], "adaline", 0.6, 0.00001)

adaline.train()

adaline.log()

perceptron = Neuron([x1, x2, x3, x4, y1, y2, y3, y4], "perceptron", 0.6, 0.00001)

perceptron.train()

perceptron.log()

print("Adaline testing")

raw, binary1 = adaline.classify_point(x1.three_d_coordinates)

print("x1 raw:", raw, "binary class:", binary1)

raw, binary2 = adaline.classify_point(x2.three_d_coordinates)

print("x2 raw:", raw, "binary class:", binary2)

raw, binary3 = adaline.classify_point(x3.three_d_coordinates)

print("x1 raw:", raw, "binary class:", binary3)

raw, binary4 = adaline.classify_point(x4.three_d_coordinates)

print("x1 raw:", raw, "binary class:", binary4)

raw, binary_y1 = adaline.classify_point(y1.three_d_coordinates)

print("y1 raw:", raw, "binary class:", binary_y1)

raw, binary_y2 = adaline.classify_point(y2.three_d_coordinates)

print("y2 raw:", raw, "binary class:", binary_y2)

raw, binary_y3 = adaline.classify_point(y3.three_d_coordinates)

print("y3 raw:", raw, "binary class:", binary_y3)

raw, binary_y4 = adaline.classify_point(y4.three_d_coordinates)

print("y4 raw:", raw, "binary class:", binary_y4)

print("Perceptron testing")

raw, binary_px1 = perceptron.classify_point(x1.three_d_coordinates)

print("x1 raw:", raw, "binary class:", binary_px1)

raw, binary_px2 = perceptron.classify_point(x2.three_d_coordinates)

print("x2 raw:", raw, "binary class:", binary_px2)

raw, binary_px3 = perceptron.classify_point(x3.three_d_coordinates)

print("x3 raw:", raw, "binary class:", binary_px3)

raw, binary_px4 = perceptron.classify_point(x4.three_d_coordinates)

print("x4 raw:", raw, "binary class:", binary_px4)

raw, binary_py1 = perceptron.classify_point(y1.three_d_coordinates)

print("y1 raw:", raw, "binary class:", binary_py1)

raw, binary_py2 = perceptron.classify_point(y2.three_d_coordinates)

print("y2 raw:", raw, "binary class:", binary_py2)

raw, binary_py3 = perceptron.classify_point(y3.three_d_coordinates)

print("y3 raw:", raw, "binary class:", binary_py3)

raw, binary_py4 = perceptron.classify_point(y4.three_d_coordinates)

print("y4 raw:", raw, "binary class:", binary_py4)

raw, binary_px5 = perceptron.classify_point([-1, -1.4, -1.5, 2])

print("Perceptron x5 raw", raw, "Binary Class ", binary_px5)

raw, binary5 = adaline.classify_point([-1, -1.4, -1.5, 2])

print("Adaline x5 raw", raw, "Binary Class ", binary5)