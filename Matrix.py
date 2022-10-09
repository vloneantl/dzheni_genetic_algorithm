import numpy as np
import csv
np.random.seed(63)

class Matrix:
    def __init__(self, path='matrix.csv', n=10, min_dist=2, max_dist=1000):
        self.__path = path
        self.__array = []
        self.n = n
        self.min_dist = min_dist
        self.max_dist = max_dist

    def generate(self):
        self.__array = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i, self.n):
                if i != j:
                    value = np.random.randint(low=self.min_dist, high=self.max_dist)
                    self.__array[i, j] = value
                if i == j:
                    self.__array[i, j] = 9999999
        for i in range(self.n):
            for j in range(self.n):
                if i < j:
                    self.__array[j, i] = self.__array[i, j]
        self.__array = self.__array.astype(int)

    def read_csv(self):
        self.__array = np.genfromtxt(self.__path, delimiter=',')

    def save_csv(self):
        f = open(self.__path, 'w')
        # create the csv writer
        writer = csv.writer(f)
        for row in self.__array:
            writer.writerow(row)
        f.close()

    def print_array(self):
        for row in self.__array:
            print(row.astype(int))

    def matrix(self):
        return self.__array.astype(int)