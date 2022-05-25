import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Converting the txt files to csv files
def txt_to_csv(file_name):
    df = pd.DataFrame(columns=['x', 'y', 'label'])
    with open(file_name, 'r') as f:
        index = 0
        for line in f:
            data = line.split(" ")
            x, y, label = data[0].strip(), data[1].strip(), data[2].strip()
            df.loc[index] = [x, y, label]
            index += 1


class Perceptron:
    def __init__(self, eta=1, epochs=20):
        self.eta = eta
        self.epochs = epochs

    def fit(self, X, y):
        # Initialize the weight vector with the size of the input and 1 for the bias with value 0: w1 = 0
        self.weights = np.zeros(X.shape[1])
        self.errors_list = []

        for epoch in range(self.epochs):
            errors = 0
            for data_points, label in zip(X, y):
                guess = self.predict(data_points)
                if guess != label:
                    errors += 1
                    # Updating the weights
                    self.weights += data_points*label
            self.errors_list.append(errors)
            if errors == 0:
                break

        print(f"Final Weights Vector: {self.weights}")
        return self

    def net_input(self, X):
        return np.dot(X, self.weights[:].T)

    def predict(self, X):
        return np.where(self.net_input(X) > 0.0, 1, -1)


def accuracy_score(y_true, y_pred):
    true_counter = 0
    size = len(y_true)
    for i in range(size):
        if y_true[i] == y_pred[i]:
            true_counter += 1

    return true_counter/size


def main():
    txt_files = ["four_circle.txt", "two_circle.txt"]
    for file_name in txt_files:
        txt_to_csv(file_name)

    # Loading the data
    file_name = "two_circle.csv"
    print("********************************************************************************")
    print(f"Loading the data from {file_name}")
    df = pd.read_csv(file_name, index_col=0)
    print(df.head())

    # Splitting the data into training and testing sets
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    print(f"Train shape: {len(X_train)}")
    print(f"Test shape: {len(X_test)}")

    epochs = [i for i in range(1, 100)]
    scores_list = {}
    for epoch in epochs:
        # Initializing and training the perceptron
        ppn = Perceptron(eta=1, epochs=epoch)
        ppn.fit(X_train, y_train)
        y_pred = ppn.predict(X_test)
        # print(f'Accuracy score for epoch {epoch}: %.2f' % accuracy_score(y_test, y_pred))
        scores_list[epoch] = accuracy_score(y_test, y_pred)

    print(f"Max score: %.3f - is with {max(scores_list, key=scores_list.get)} epochs" % max(scores_list.values()))
    print(f"Min score: %.3f - is with {min(scores_list, key=scores_list.get)} epochs" % min(scores_list.values()))
    print(f"Average score: %.3f" % float(sum(scores_list.values())/len(scores_list.values())))
    print(f"Errors list: {ppn.errors_list}")
    print(f"scores list len: {scores_list}")

    # Plotting the final test results
    print(f"y pred: {y_pred}")
    plt.scatter(X_test[:, 0], X_test[:, 1], color=["red" if y_pred[i] == 1 else "blue" for i in range(len(y_pred))])
    plt.title("Perceptron - " + file_name.split(".")[0] + " - Final test results")
    plt.savefig(f"images/{file_name.split('.')[0]}_final_results.png")
    plt.show()

    # Plotting the errors
    plt.plot(range(1, len(ppn.errors_list) + 1), ppn.errors_list)
    plt.title("Perceptron - " + file_name.split(".")[0] + " - Error")
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.savefig(f"images/{file_name.split('.')[0]}_error.png")
    plt.show()

    print("********************************************************************************\n\n")


if __name__ == '__main__':
    main()

