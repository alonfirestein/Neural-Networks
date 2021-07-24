import matplotlib.pyplot as plt
import Kohonen
import numpy as np
from random import random
import cv2
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


# 15x15 Mesh Grid
def initNeuronsInMesh(neurons_size):
    neurons = np.zeros((neurons_size, 2))
    rows, cols = 15, 15
    x, y = 0.33, 0.0
    step_size = 0.03
    for i in range(rows):
        for j in range(cols):
            neuron = np.array([x, y])
            neurons[i * cols + j] = neuron
            x += step_size
        x = 0.33
        y += step_size

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
    res[:, 0], res[:, 1] = x, y
    return res


def create_points_for_hand(image, size):
    original_points = np.zeros((size, 2))
    normalized_points = np.zeros((size, 2))
    for i in range(size):
        flag = True
        while flag:
            x, y = np.random.randint(0, 1152), np.random.randint(0, 648)
            if image[y, x] == 255:
                original_points[i] = np.array([y, x])
                normalized_points[i] = np.array([(1152-x)/1152, (648-y)/648])
                flag = False

    return original_points, normalized_points


def get_points_for_second_hand(image2, points):
    arr = np.zeros((len(points), 2))
    j = 0
    for i in range(len(points)):
        y, x = points[i]
        x, y = int(x), int(y)
        if image2[y, x] == 255:
            arr[j] = np.array([(1152-x)/1152, (648-y)/648])
            j += 1

    return arr[:j, :]


def run_kohonen(shape_type, epochs, isUniform, image1=None, image2=None):
    fig, ax = plt.subplots()

    if shape_type.lower() == "square":
        sample = initSquareSample(250)
        size_of_neurons = 15
        if not isUniform:
            sample = initNonUniformPoints(500)

    if "hand" in shape_type.lower():
        original, sample = create_points_for_hand(image1, 25000)
        size_of_neurons = 15*15

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

    if "hand" in shape_type.lower():
        neurons = initNeuronsInMesh(size_of_neurons)

    else:
        neurons = initNeuronsInLine(size_of_neurons)

    Kohonen.fit(sample, neurons, epochs, radius)
    plt.scatter(sample[:, 0], sample[:, 1], color='green', marker='x', label='points')
    plt.scatter(neurons[:, 0], neurons[:, 1], color='red', marker='o', label='neurons')

    # If we cut of a finger, we will use the trained data from the hand with 4 fingers and then let the neurons
    # rearrange themselves again to fit within the borders of the new hand with only 3 fingers.
    if shape_type.lower() == "hand3":
        plt.show()
        sample = get_points_for_second_hand(image2, original)
        Kohonen.fit(sample, neurons, epochs, radius)
        plt.scatter(sample[:, 0], sample[:, 1], color='green', marker='x', label='points')
        plt.scatter(neurons[:, 0], neurons[:, 1], color='red', marker='o', label='neurons')

    if "hand" not in shape_type.lower():
        plt.plot(neurons[:, 0], neurons[:, 1], color='red', linewidth=1.0)

    plt.show()


if __name__ == '__main__':
    image1 = cv2.imread("/Users/alon/PycharmProjects/Adaline/4_finger_hand.jpeg", 0)
    image2 = cv2.imread("/Users/alon/PycharmProjects/Adaline/3_finger_hand.jpeg", 0)
    run_kohonen(shape_type="hand", epochs=50, isUniform=False, image1=image1, image2=None)
