import random
import numpy as np
from itertools import permutations
np.random.seed(77)

class Chromosome:
    def __init__(self, n=10, arr=[]):
        self.n = n
        self.__array = arr
        if not len(arr):
            self.__generate()

    def __generate(self):
        arr = list(range(1, self.n))
        np.random.shuffle(arr)
        self.__array = [0] + arr

    def print_arr(self):
        print(self.__array)

    def get_chromosome(self):
        return self.__array

    def fitness(self, matrix):
        '''
        Подсчет функции приспособленности
        matrix : Matrix
        :return: float
        '''
        sum = 0
        for i in range(self.n-1):
            print(matrix[self.__array[i], self.__array[i+1]])
            sum+=matrix[self.__array[i], self.__array[i+1]]
        sum+=matrix[self.__array[-1], self.__array[0]]
        return sum

from Matrix import Matrix
matr1 = Matrix(n=10, min_dist=70, max_dist=100, trivial_definition=1)
matr1.generate()
matr1.save_csv()
matr1.read_csv()
matr1.print_array()
print(Chromosome(10,[0,1,2,3,4,5,6,7,8,9]).fitness(matr1.matrix()))