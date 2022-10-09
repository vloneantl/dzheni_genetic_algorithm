import random
import numpy as np
from itertools import permutations
#np.random.seed(1337)

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
        Подсчет критерия остановки, функция приспособленности
        matrix : Matrix
        :return: float
        '''
        sum = 0
        for i in range(self.n-1):
            sum+=matrix[self.__array[i],self.__array[i+1]]
        sum+=matrix[self.__array[-1],self.__array[0]]
        return sum
