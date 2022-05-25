import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def txt_to_csv(file_name):
    """
    Converting the txt files to csv files
    :param file_name:
    :return:
    """
    df = pd.DataFrame(columns=['x', 'y', 'label'])
    with open(file_name, 'r') as f:
        index = 0
        for line in f:
            data = line.split(" ")
            x, y, label = data[0].strip(), data[1].strip(), data[2].strip()
            df.loc[index] = [x, y, label]
            index += 1


def get_label_from_rule(rule, point):
    rule_label = rule[2]
    # Point[0] - X of point,  Point[1] - Y of point
    # Rule[0] - the first point of the rule,  Rule[1] - the second point of the rule
    data = ((point[0] - rule[0][0]) * (rule[1][1] - rule[0][1])) - \
        ((point[1] - rule[0][1]) * (rule[1][0] - rule[0][0]))

    # If the rule label was correctly predicted, return 1
    correctly_predicted_label = (rule_label == 1 and data >= 0) or (rule_label == -1 and data < 0)
    if correctly_predicted_label:
        return 1
    # Else, if it was incorrect, return -1
    return -1


def calculate_error(points, labels, combinationOfRules, combinationOfRulesWeight):
    """
    Calculates the error in our training/testing sets
    :param points: The list of points in our training/testing sets
    :param labels: The list of labels of our training/testing sets points
    :param combinationOfRules: Our combination of rules
    :param combinationOfRulesWeight: The weights of our combination of rules
    :return: The average error for our train/test set per k
    """
    totalErr = 0
    for i in range(len(points)):
        totalSum = 0
        for j in range(len(combinationOfRules)):
            guessLabel = get_label_from_rule(combinationOfRules[j], points[i])
            totalSum += guessLabel * combinationOfRulesWeight[j]
        if totalSum < 0 and labels[i] != -1:
            totalErr += 1
        if totalSum >= 0 and labels[i] != 1:
            totalErr += 1

    return totalErr / len(points)


def plot_results(train_error, test_error):
    """
    Plotting the average train and test error for each k = 1 to 8
    :param train_error: list of average train error per k
    :param test_error: list of average test error per k
    :return:
    """
    plt.plot(range(1, len(train_error) + 1), train_error, label='Train error')
    plt.plot(range(1, len(test_error) + 1), test_error, label='Test error')
    plt.title("AdaBoost - Error")
    plt.xlabel('# of K')
    plt.ylabel('Average Error')
    plt.legend()
    plt.savefig(f"images/adaboost_error.png")
    plt.show()


class AdaBoost:
    def __init__(self, X_train, X_test, y_train, y_test, k, epochs):
        self.k = k
        self.epochs = epochs
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.size = len(X_train)
        self.train_error_avg = [0 for _ in range(8)]
        self.test_error_avg = [0 for _ in range(8)]

    def fit(self):
        for epoch in range(self.epochs):
            print(f"\n***** Starting Epoch Number {epoch+1} *****")
            pointsWeight = [1/self.size for _ in range(self.size)]
            bestRules, bestRulesWeight = [], []
            hypothesis_set = self.init_hypothesis_set()

            for i in range(self.k):
                bestRuleValue, bestRuleIndex = self.find_best_rule(pointsWeight, hypothesis_set)
                # print(f"K #{i+1} - Best Rule: {bestRuleValue} - of index: {bestRuleIndex}")
                bestRules.append(hypothesis_set[bestRuleIndex])
                alphaT = 0.5 * np.log((1.0 - bestRuleValue) / bestRuleValue)
                # print(f"K #{i+1} - alphaT = {alphaT}\n")
                bestRulesWeight.append(alphaT)
                self.update_weight(alphaT, pointsWeight, hypothesis_set[bestRuleIndex])

            self.get_TrainTest_error(bestRules, bestRulesWeight)
            print(f"***** Finished Epoch Number {epoch+1} *****\n")

    def init_hypothesis_set(self):
        """
        Each pair of points can define a line that passes through the two points. The set of all such lines is our
        hypothesis set, that is our set of rules.
        This function initializes this set of rules.
        :return: Our hypothesis set
        """
        hypothesis_set = []
        for i in range(self.size):
            for j in range(i + 1, self.size):
                if i != j:
                    # Appending two rules for each pair of points, one for each label
                    hypothesis_set.append([self.X_train[i], self.X_train[j], 1])
                    hypothesis_set.append([self.X_train[i], self.X_train[j], -1])
        return hypothesis_set

    def find_best_rule(self, pointsWeight, hypothesis_set):
        """
        Function to find the best rule (its value and index from hypothesis set) from the large set of rules
        in each k= 1 to 8.
        :param pointsWeight: List of weight of points
        :param hypothesis_set: Our hypothesis set (Our set of rules)
        :return: The value and index of the best rule found
        """
        totalNumOfRules, bestRuleValue, bestRuleIndex = len(hypothesis_set), sys.maxsize, 0
        # For each rule, if rule sum is lower than we take that one as our current best rule
        for i in range(totalNumOfRules):
            hypothesis_sum = self.get_hypothesis_sum(pointsWeight, hypothesis_set[i])
            if hypothesis_sum < bestRuleValue:
                bestRuleValue = hypothesis_sum
                bestRuleIndex = i

        return bestRuleValue, bestRuleIndex

    def get_hypothesis_sum(self, pointsWeight, rule):
        """
        Getting the sum of all the rules in our hypothesis set
        :param pointsWeight: List of weight of points
        :param rule: Current Rule
        :return:
        """
        hypothesis_sum = 0
        # For each point in the training data points we calculate the total rule sum
        for i in range(self.size):
            currentLabel = get_label_from_rule(rule, self.X_train[i])
            if currentLabel != self.y_train[i]:
                hypothesis_sum += pointsWeight[i]
        return hypothesis_sum

    def update_weight(self, bestRuleWeight, pointsWeight, rule):
        """
        Updating the weights - If the model incorrectly predicted the label, the weights will increase for the next
        round, and if the model correctly predicts the label, the weights will decrease for the next round.
        :param bestRuleWeight: The weight of the best rule found
        :param pointsWeight: List of weights of the points
        :param rule: Current rule
        :return:
        """
        for i in range(self.size):
            label = get_label_from_rule(rule, self.X_train[i])
            pointsWeight[i] = pointsWeight[i] * np.exp(-(bestRuleWeight * label * self.y_train[i]))

        for i in range(self.size):
            pointsWeight[i] = pointsWeight[i] / sum(pointsWeight)

    def get_TrainTest_error(self, bestRules, bestRulesWeight):
        """
        Get the average error of the train and test set for each k= 1 to 8
        :param bestRules: The list of of the 8 best rules found in each iteration
        :param bestRulesWeight: The list of weights of the 8 best rules of each iteration
        :return:
        """
        combinationOfRules = []
        combinationOfRulesWeight = []

        for i in range(self.k):
            combinationOfRules.append(bestRules[i])
            combinationOfRulesWeight.append(bestRulesWeight[i])
            test_error = calculate_error(self.X_test, self.y_test, combinationOfRules, combinationOfRulesWeight)
            train_error = calculate_error(self.X_train, self.y_train, combinationOfRules, combinationOfRulesWeight)
            self.train_error_avg[i] += train_error
            self.test_error_avg[i] += test_error

    def get_results(self):
        """
        Writes the train and test error averages into a txt file named "adaboost_results.txt"
        :return:
        """
        result = ""
        for i in range(self.k):
            result += "Combine {} rules - Train error avg: {:.5f}\n".format((i+1), self.train_error_avg[i]/self.epochs)
            result += "Combine {} rules - Test error avg: {:.5f}\n\n".format((i+1), self.test_error_avg[i]/self.epochs)

        with open("adaboost_results.txt", 'w') as file:
            file.write(result)
        print(result)


def main():
    file_name = "four_circle.txt"
    txt_to_csv(file_name)

    # Loading the data
    file_name = "four_circle.csv"
    print("********************************************************************************")
    print(f"Loading the data from {file_name}")
    df = pd.read_csv(file_name, index_col=0)
    print(df.head())

    # Splitting the data into training and testing sets
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    print(f"Train shape: {len(X_train)}")
    print(f"Test shape: {len(X_test)}")

    # For each k=1,...,8, compute the empirical error... (from the assignment)
    adaboost = AdaBoost(X_train, X_test, y_train, y_test, k=8, epochs=100)
    adaboost.fit()
    adaboost.get_results()
    plot_results(adaboost.train_error_avg, adaboost.test_error_avg)


if __name__ == '__main__':
    main()
