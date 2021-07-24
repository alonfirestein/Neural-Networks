import numpy as np


# Calculating the distance between two points
def euclideanDistance(point, neuron):
    return np.sqrt(np.power(point[0] - neuron[0], 2) + np.power(point[1] - neuron[1], 2))


# Choosing best matching unit to the current point
def BMU(point, neurons):
    min_distance = np.inf
    min_distance_index = -1
    for index in range(len(neurons) - 1):
        current_distance = euclideanDistance(point, neurons[index])
        if current_distance < min_distance:
            min_distance = current_distance
            min_distance_index = index

    return min_distance_index


# Topological neighborhood
def topologicalNeighborhood(winner_neuron, neuron, sigma):
    distance = euclideanDistance(winner_neuron, neuron)
    return np.exp(- (np.power(distance, 2) / (2 * np.power(sigma, 2))))


# Updating the neurons location
def updateNeuronLocation(point, neuron, alpha, topo):
    return neuron + alpha * topo * (point - neuron)


# Updating the sigma by the epoch number
def updateSigma(sigma, iteration, lambda_start):
    return sigma * np.exp(-iteration / lambda_start)


# Updating the alpha by the epoch number
def updateLearningRate(alpha, iteration, lambda_start):
    return alpha * np.exp(-iteration / lambda_start)


def fit(points, neurons, epochs, radius):
    learning_rate = 0.01
    sigma_start = radius / 2 + 0.0001
    lambda_start = epochs / np.log(epochs)

    # Train each point by the number of epochs:
    for iteration in range(epochs):

        # Update the learning rate and neighborhood size after every epoch
        new_lr = updateLearningRate(learning_rate, iteration, epochs)
        # Sigma => neighborhood size
        sigma = updateSigma(sigma_start, iteration, lambda_start)

        for point in points:
            # Find the "winner" neuron
            winner_neuron = BMU(point, neurons)
            # For each neuron
            for neighbor_index in range(len(neurons)):
                distance = euclideanDistance(neurons[winner_neuron], neurons[neighbor_index])
                if distance < sigma:
                    topo = topologicalNeighborhood(neurons[winner_neuron], neurons[neighbor_index], sigma)
                    neurons[neighbor_index] = updateNeuronLocation(point, neurons[neighbor_index], new_lr, topo)

