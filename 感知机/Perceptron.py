import matplotlib.pyplot as plt
import numpy as np


class PerceptronDual:
    def __init__(self, train_data):
        data_len = len(train_data)
        self.a = [0] * data_len
        self.b = 0
        gram = []
        for x_i in train_data:
            gram_row = []
            for x_j in train_data:
                gram_row.append(np.dot(x_i[0], x_j[0]))

            gram.append(gram_row)
        self.gram = np.array(gram)
        self.train_data = train_data
        self.w = 0

    def train(self):
        for i in range(len(self.train_data)):
            w = 0
            gram_row = self.gram[:, i]
            for j in range(len(self.a)):
                w += (self.a[j] * self.train_data[j][1] * gram_row[j])
            if self.train_data[i][1] * (w + self.b) <= 0:
                self.a[i] += 1
                self.b += self.train_data[i][1]
                print(self.b)
                return False
        self.w = 0
        for j in range(len(self.a)):
            self.w += (self.a[j] * self.train_data[j][0])
        return True


class Perceptron:

    def __init__(self):
        self.w = np.array([0, 0])
        self.b = 0
        self.alpha = 1

    def filter_error(self, train_data):
        for item in train_data:
            result = (np.dot(item[0], self.w) + self.b) * item[1]
            if result <= 0:
                return item
        return None

    def adjust_param(self, item_data):
        self.w = self.w + self.alpha * item_data[1] * item_data[0]
        self.b = self.b + self.alpha * item_data[1]
        print(self.w, self.b)

    def get_line_point(self):
        x_points = [0, -self.b / self.w[0]]
        y_points = [-self.b / self.w[1], 0]
        return x_points, y_points


def perceptron_normal():
    plt.figure(figsize=(8, 6), dpi=80)
    # plt.ion()
    x_positive = [4, 1.5, 2.5, 3, 2]
    y_positive = [3.3, 5, 4, 3.2, 3.8]
    x_negative = [0.5, 2.5, 1, 2, 1.5]
    y_negative = [2.5, 2.9, 1.5, 2.6, 3.3]

    train_data = []
    for nIndex in range(len(x_positive)):
        train_data.append([np.array([x_positive[nIndex], y_positive[nIndex]]), 1])
    for nIndex in range(len(x_negative)):
        train_data.append([np.array([x_negative[nIndex], y_negative[nIndex]]), -1])
    perceptron = Perceptron()
    while True:
        plt.cla()
        plt.scatter(x_positive, y_positive, c="r", marker="o")
        plt.scatter(x_negative, y_negative, c="b", marker="v")
        filter_data = perceptron.filter_error(train_data)
        if filter_data is None:
            break
        perceptron.adjust_param(filter_data)
        x_points, y_points = perceptron.get_line_point()

        plt.plot(x_points, y_points)
        plt.pause(0.5)
    plt.show()


def perceptron_dual():
    perceptron = PerceptronDual([[np.array([3,3]),1],[np.array([4,3]),1],[np.array([1,1]),-1]])
    while True:
        if perceptron.train() is True:
            break


if __name__ == "__main__":
    perceptron_dual()
