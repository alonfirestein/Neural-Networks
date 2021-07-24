import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def create_dataset(size, n_size, condition):
    x_list, y_list, expected_list = list(), list(), list()
    for i in range(size):
        x = random.randint(-n_size, n_size) / n_size
        y = random.randint(-n_size, n_size) / n_size
        x_list.append(x)
        y_list.append(y)
        if condition == 'A':
            expected_result = np.where((x > 0.5 and y > 0.5), 1, -1)
            expected_list.append(expected_result)
        else:
            expected_result = np.where(((x**2 + y**2 >= 0.5) and (x**2 +y**2 <= 0.75)), 1, -1)
            expected_list.append(expected_result)

    data = {"X": x_list,
            "Y": y_list,
            "Expected_Result": expected_list}
    df = pd.DataFrame(data, columns=["X", "Y", "Expected_Result"])
    return df


class Adaline(object):

    def __init__(self, learning_rate=0.001, n_iter=20):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit(self, X, y):
        self.weight = np.zeros(1 + X.shape[1])
        self.error_list = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.weight[1:] = self.weight[1:] + self.learning_rate * X.T.dot(errors)
            self.weight[0] = self.weight[0] + self.learning_rate * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.error_list.append(cost)

        return self

    def net_input(self, X):
        return np.dot(X, self.weight[1:]) + self.weight[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)


def accuracy_score(y_true, y_pred):
    true_counter = 0
    size = len(y_true)
    for i in range(size):
        if y_true[i] == y_pred[i]:
            true_counter += 1

    return true_counter/size


if __name__ == '__main__':

    # Creating the dataset
    a = create_dataset(1000, 10000, "A")
    X = a.drop("Expected_Result", axis=1)
    y = a["Expected_Result"]

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    X_train, X_test, y_train, y_test = X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

    # Creating and initializing the Adaline model we built
    adaline = Adaline(n_iter=15, learning_rate=0.0001)

    # Training the model we built
    adaline.fit(X, y)
    y_pred = adaline.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print("Accuracy Score: %f" % score)

    """
    Accuracy Score Pie Chart Plot Using Matplotlib
    """
    # labels = 'Correct', 'Incorrect'
    # sizes = [score, 1-score]
    # fig1, ax1 = plt.subplots()
    # ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=45)
    # ax1.axis('equal')
    # plt.title("Accuracy Score - Part A")
    # plt.show()

    """
    SSE Per Epoch Plot Using Matplotlib 
    """
    # plt.plot(range(1, len(adaline.error_list) + 1), adaline.error_list, marker='o', color='purple')
    # plt.title("Part B: Bigger Data Size")
    # plt.xlabel('Epochs')
    # plt.ylabel('Sum-squared-error')
    # plt.show()
