from math import exp
import numpy as np
import random
from random import seed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from Adaline import Adaline


def create_dataset(size, n_size, condition):
    dataset = list()
    for i in range(size):
        x = random.randint(-n_size, n_size) / n_size
        y = random.randint(-n_size, n_size) / n_size
        if condition == 'A':
            if x > 0.5 and y > 0.5:
                expected_result = 1
            else:
                expected_result = -1
        else:
            if (x ** 2 + y ** 2 >= 0.5) and (x ** 2 + y ** 2 <= 0.75):
                expected_result = 1
            else:
                expected_result = -1
        dataset.append([x, y, expected_result])

    return dataset


# Initializing the network
def network_init(n_inputs, n_hidden, n_outputs):
    network = list()
    # one hidden layer of n_hidden neurons
    hidden_layer = [{'weights': [random.random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)

    # one neuron of output for each class (in our case: 2 output neurons)
    # each neuron gives us the probability to be classified by the class which the neuron represents
    output_layer = [{'weights': [random.random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# Calculating the neuron activation for an input.
# Its a net_input function, like in Adaline (sigma over input and weight)
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# Sigmoid function on neuron activation
def sigmoid(activation):
    return 1.0 / (1.0 + exp(-activation))


# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = list()
        # for each neuron we compute the activation function, then we give the result to the next level of neurons
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = sigmoid(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs  # the new input for the next layer
    return inputs


# Calculating the derivative of an neuron output
def derivative(output):
    return output * (1.0 - output)


# Back propagate error and store it in the neurons
def backward_propagate(network, expected):
    # return back from the output layer to the input layer
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        # If i is the output layer
        if i == len(network) - 1:
            for j in range(len(layer)):
                neuron = layer[j]
                # the error this the difference between the the expected and the actual output
                error = expected[j] - neuron['output']
                errors.append(error)

        # If i is the hidden layer
        else:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)

        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * derivative(neuron['output'])


# Updating the neural networks weights with error
def update_weights(network, row, eta):
    for i in range(len(network)):
        inputs = row[:-1]
        # If i is the output layer
        if i == 1:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += np.multiply(eta, np.multiply(neuron['delta'], inputs[j]))
            neuron['weights'][-1] += np.multiply(eta, neuron['delta'])


# Train a network for a fixed number of epochs
def train(network, dataset, eta, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in dataset:
            outputs = forward_propagate(network, row)  # first move forward
            # array of zero except in the index of the right class - its represent te probability
            expected = np.zeros(n_outputs)
            expected[row[-1]] = 1
            # accumulated error pf all the data
            sum_error += sum([np.power((expected[i] - outputs[i]), 2) for i in range(len(expected))])
            # make the backward move
            backward_propagate(network, expected)
            # then update the weights
            update_weights(network, row, eta)
        print('-> Epoch=%d, ETA=%.3f, Error=%.3f' % (epoch, eta, sum_error))


def plot_NN(network, dataset):
    """

    :param network:
    :param dataset:
    :return: a plot of the neural network for each neuron
    """
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    plot_index = 1
    for layer in network:
        x_list, y_list, result_list = list(), list(), list()
        for neuron in layer:
            for row in dataset:
                activation = activate(neuron['weights'], row)
                if sigmoid(activation) >= 0.5:
                    result = 'red'
                else:
                    result = 'blue'
                x_list.append(row[0])
                y_list.append(row[1])
                result_list.append(result)
            plt.subplot(2, 3, plot_index)
            plt.scatter(x_list, y_list, color=result_list)
            red_patch, blue_patch = mpatches.Patch(color='red', label='1'), mpatches.Patch(color='blue', label='0')
            if plot_index == 2:
                plt.title("Hidden Layers")
            if plot_index == 5:
                plt.title("Output Layers")
            plot_index += 1
            plt.legend(handles=[red_patch, blue_patch])
    plt.tight_layout()
    plt.show()


def predict(network, test):
    """Return class label after unit step"""
    result = list()
    for row in test:
        ones, zeros = forward_propagate(network, row)
        if ones >= 0.5:
            result.append(1)
        else:
            result.append(-1)
    return result


def accuracy_score(y_true, y_pred):
    true_counter = 0
    size = len(y_true)
    for i in range(size):
        if y_true[i] == y_pred[i]:
            true_counter += 1

    return true_counter / size


################################ PART D ####################################

def FeedAdaline(network, train):
    """

    :param network:
    :param train: the training set
    :return: a list of outputs from the output layer to feed the Adaline neurons
    """
    result = list()
    for row in train:
        result.append(forward_propagate(network, row))
    return result


def concate_Adaline_BackProg():
    """
    :return: a neural network with Adaline neuron as output layer
    """

    # First create the data set, split it into test and train, then fit the neural network
    seed(1)
    num_of_points = 1000
    testing_size, training_size = int(num_of_points * 0.33), int(num_of_points * 0.67)
    dataset = create_dataset(num_of_points, 100, "B")
    train_set = dataset[:training_size]
    test_set = dataset[testing_size:]

    n_inputs = len(train_set[0]) - 1
    n_outputs = len(set([row[-1] for row in train_set]))
    network = network_init(n_inputs, 3, n_outputs)
    train(network, train_set, 0.4, 20, n_outputs)
    # now the neural network is fitted

    # Initializing Adaline
    adaline = Adaline(n_iter=15, learning_rate=0.0001)

    # We feed the Adaline by the results of the output layer
    X = np.array(FeedAdaline(network, np.array(train_set)[:, :2]))
    y = np.array(train_set)[:, 2]

    adaline.fit(X, y)

    X_test = np.array(test_set)[:, :2]
    y_true = np.array(test_set)[:, 2]

    y_pred = adaline.predict(X_test)
    score = accuracy_score(y_true, y_pred)
    print("Accuracy Score: %f" % score)


if __name__ == '__main__':
    # Back-Propagation
    seed(1)
    num_of_points = 1000
    testing_size, training_size = int(num_of_points * 0.3), int(num_of_points * 0.7)
    dataset = create_dataset(num_of_points, 100, "B")
    train_set = dataset[:training_size]
    test_set = dataset[testing_size:]

    n_inputs = len(train_set[0]) - 1
    n_outputs = len(set([row[-1] for row in train_set]))
    network = network_init(n_inputs, 3, n_outputs)
    train(network, train_set, 0.4, 20, n_outputs)

    y_true, y_pred = np.array(test_set)[:, 2], predict(network, test_set)
    print("Accuracy Score: ", accuracy_score(y_true, y_pred))
    plot_NN(network, train_set)

    concate_Adaline_BackProg()
