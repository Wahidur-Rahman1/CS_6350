#!/usr/bin/env python
# coding: utf-8

# In[13]:


import csv
import random

class Perceptron:
    def __init__(self, T, learning_rate):
        self.T = T
        self.learning_rate = learning_rate
        self.weight = [0, 0, 0, 0]
        self.bias = 0

    @staticmethod
    def read_dataset(filename):
        with open(filename, 'r') as f:
            r = csv.reader(f)
            dataset = [[float(i) for i in line[:-1]] + [1 if line[-1] == '1' else -1]
                       for line in r]
        return dataset

    @staticmethod
    def get_sign(n):
        return 1 if n > 0 else -1
        print("Standard Perceptron")
    def train(self, dataset):
        for i in range(self.T):
            random.shuffle(dataset)
            error_count = sum(self.update_weights(data) for data in dataset)
            print(f'epoch {i}, error rate: {error_count / len(dataset):f}')
        print("STANDARD PERCEPTRON")    
        print("Weights and Bias: ", self.weight, self.bias)

    def update_weights(self, data):
        x = data[:-1]
        y = data[-1]
        predict = self.get_sign(sum(w*x_i for w, x_i in zip(self.weight, x)) + self.bias)
        if y * predict <= 0:
            self.weight = [w + self.learning_rate * y * x_i for w, x_i in zip(self.weight, x)]
            self.bias += self.learning_rate * y
            return 1
        return 0

    def test(self, dataset):
        error_count = sum(self.predict(data) != data[-1] for data in dataset)
        print(f'Average error: {error_count / len(dataset):f}')

    def predict(self, data):
        x = data[:-1]
        y = data[-1]
        result = sum(w*x_i for w, x_i in zip(self.weight, x)) + self.bias
        return self.get_sign(result)


def run_perceptron(train_file, test_file, T, learning_rate):
    p = Perceptron(T, learning_rate)
    train_data = p.read_dataset(train_file)
    test_data = p.read_dataset(test_file)
    p.train(train_data)
    p.test(test_data)


if __name__ == '__main__':
    T = 10
    learning_rate = 0.01
    train_file = 'train.csv'
    test_file = 'test.csv'
    run_perceptron(train_file, test_file, T, learning_rate)

    
######################################################    

class Classifier:
    def __init__(self, train_file, test_file, learning_rate=0.01, epochs=10):
        self.train_file = train_file
        self.test_file = test_file
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = [0.0] * 4
        self.bias = 0.0

    @staticmethod
    def load_data(filename):
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            data = [[float(i) for i in line[:-1]] + [1 if line[-1] == '1' else -1] for line in reader]
        return data

    @staticmethod
    def sign(n):
        return 1 if n > 0 else -1

    def train(self):
        train_data = self.load_data(self.train_file)
        for epoch in range(self.epochs):
            random.shuffle(train_data)
            errors = 0
            for data in train_data:
                x = data[:-1]
                y = data[-1]
                prediction = self.sign(sum(w * x_i for w, x_i in zip(self.weights, x)) + self.bias)
                if y * prediction <= 0:
                    errors += 1
                    for i in range(len(self.weights)):
                        self.weights[i] += self.learning_rate * y * x[i]
                    self.bias += self.learning_rate * y
            error_rate = errors / len(train_data)
            print(f'epoch {epoch}, error rate: {error_rate}')
        print("AVERAGED PERCEPTRON")    
        return self.weights, self.bias

    def test(self):
        test_data = self.load_data(self.test_file)
        errors = 0
        for data in test_data:
            x = data[:-1]
            y = data[-1]
            prediction = self.sign(sum(w * x_i for w, x_i in zip(self.weights, x)) + self.bias)
            if y * prediction <= 0:
                errors += 1
        error_rate = errors / len(test_data)
        print(f'Average error: {error_rate}')

    def run(self):
        self.train()
        self.test()


if __name__ == '__main__':
    classifier = Classifier('train.csv', 'test.csv')
    classifier.run()
######################################################    
class Classifier:
    def __init__(self, train_file, test_file, learning_rate=0.01, epochs=10):
        self.train_file = train_file
        self.test_file = test_file
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.biases = []
        self.counts = []
        self.margin = -1

    @staticmethod
    def load_data(filename):
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            data = [[float(i) for i in line[:-1]] + [1 if line[-1] == '1' else -1] for line in reader]
        return data

    @staticmethod
    def sign(n):
        return 1 if n > 0 else -1

    def train(self):
        train_data = self.load_data(self.train_file)
        weight = [0.0] * 4
        bias = 0.0
        c = 0

        for epoch in range(self.epochs):
            random.shuffle(train_data)
            errors = 0
            for data in train_data:
                x = data[:-1]
                y = data[-1]
                prediction = self.sign(sum(w * x_i for w, x_i in zip(weight, x)) + bias)
                if y * prediction <= 0:
                    errors += 1
                    for i in range(len(weight)):
                        weight[i] += self.learning_rate * y * x[i]
                    bias += self.learning_rate * y
                    self.weights.append(weight[:])  # copy the list to avoid reference issues
                    self.biases.append(bias)
                    self.counts.append(c)
                    c = 1
                    self.margin += 1
                else:
                    c += 1
            error_rate = errors / len(train_data)
            print(f'epoch {epoch}, error rate: {error_rate}')

        print("Weight and Bias:")
        for i in range(len(self.weights)):
            print(self.weights[i] + [self.biases[i]], self.counts[i])
        print(self.margin + 1)

    def test(self):
        test_data = self.load_data(self.test_file)
        errors = 0
        for data in test_data:
            x = data[:-1]
            y = data[-1]
            res = sum(c * self.sign(sum(w * x_i for w, x_i in zip(weight, x)) + bias) for weight, bias, c in zip(self.weights, self.biases, self.counts))
            prediction = self.sign(res)
            if y * prediction <= 0:
                errors += 1
        error_rate = errors / len(test_data)
        print("VOTED PERCEPTRON") 
        print(f'Average error: {error_rate}')

    def run(self):
        self.train()
        self.test()


if __name__ == '__main__':
    classifier = Classifier('train.csv', 'test.csv')
    classifier.run()

    

