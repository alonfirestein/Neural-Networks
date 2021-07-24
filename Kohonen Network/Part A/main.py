import matplotlib.pyplot as plt
import Kohonen
import numpy as np
from random import random
radius = 0.25
centerX = 0.5
centerY = 0.5


def initNeuronsInLine(neurons_size):
    neurons = np.zeros((neurons_size, 2))
    step = 1 / len(neurons)
    start_point = 0
    for i in range(neurons_size):
        neurons[i] = np.array([start_point, 0.5])
        start_point += step

    return neurons


def initSquareSample(size):
    return np.random.uniform(0, 1, (size, 2))


def initDonutSample(size, big_radius, small_radius):
    donut = np.zeros((size, 2))

    for i in range(size):
        x = 0
        y = 0
        while x < small_radius and y < small_radius:
            dist = np.sqrt(random() * (big_radius**2-small_radius**2) + small_radius**2)
            theta = 360 * random()
            x = 0.5 + dist * np.cos(theta)
            y = 0.5 + dist * np.sin(theta)
        donut[i] = np.array([x, y])

    return donut


def initNonUniformPoints(size):
    x = np.random.normal(loc=0.5, scale=0.2, size=(size))
    y = np.random.uniform(0,1, size)
    res = np.zeros((size,2))
    res[:, 0] , res[: ,1] = x, y
    return res


def run_kohonen(shape_type, epochs, isUniform):
    fig, ax = plt.subplots()

    if shape_type.lower() == "square":
        sample = initSquareSample(250)
        size_of_neurons = 15
        if not isUniform:
            sample = initNonUniformPoints(500)

    else:
        sample = initDonutSample(500, radius*2, radius)
        size_of_neurons = 30
        theta = np.linspace(0, 2 * np.pi, 500)
        big_radius = radius*2
        inner1 = centerX + radius * np.cos(theta)
        inner2 = centerY + radius * np.sin(theta)
        outer1 = centerX + big_radius * np.cos(theta)
        outer2 = centerY + big_radius * np.sin(theta)
        ax.plot(inner1, inner2, color='blue')
        ax.plot(outer1, outer2, color='blue')

    neurons = initNeuronsInLine(size_of_neurons)
    Kohonen.fit(sample, neurons, epochs, radius)

    plt.scatter(sample[:, 0], sample[:, 1], color='green', marker='x', label='points')
    plt.scatter(neurons[:, 0], neurons[:, 1], color='red', marker='o', label='neurons')
    plt.plot(neurons[:, 0], neurons[:, 1], color='red', linewidth=1.0)
    plt.show()


if __name__ == '__main__':
    run_kohonen(shape_type="square", epochs=100, isUniform=False)
