from chromosome import Chromosome
import numpy as np
np.random.seed(63)

class Population:
    def __init__(self, matrix, m=5, n=10, pc=0.7, pm=0.3):
        '''

        :param m: количество особей в популяции
        :param n: количество городов
        ????????:param k: количество родителей
        '''
        self.m = m
        self.n = n
        self.pc = pc #вероятность кроссовера
        self.__list = []
        self.generate_chromosomes()
        self.matrix = matrix
        #self.sort_list()
        self.pm = pm

    def generate_chromosomes(self):
        while len(self.__list) < self.m:
            chrome = Chromosome(self.n)
            if chrome in self.__list:
                continue
            self.__list.append(chrome)

    def get_list(self):
        return self.__list

    def set_list(self, l):
        self.__list = l

    def print_list(self):
        for item in self.__list:
            print(item.get_chromosome(), item.fitness(self.matrix.matrix()))

    def crossover(self, parent1, parent2):
        if np.random.random() > self.pc:
            return parent1, parent2
        parent1, parent2 = parent1.get_chromosome(), parent2.get_chromosome()
        swap_index = np.random.randint(2, self.n-1)
        chrome1 = parent2[:swap_index]
        chrome2 = parent1[:swap_index]
        for i in range(0, self.n):
            if parent1[i] not in chrome1:
                chrome1.append(parent1[i])
            if parent2[i] not in chrome2:
                chrome2.append(parent2[i])
        self.__list = self.__list + [Chromosome(self.n, chrome1), Chromosome(self.n, chrome2)]
        #self.__sort_list()

    def sort_list(self):
        self.__list = sorted(self.__list, key=lambda x: x.fitness(self.matrix.matrix()))
        return self

    def parents_selection(self):
        parents_reverse_fitness = [1/item.fitness(self.matrix.matrix()) for item in self.__list]
        parents_prob = [item / sum(np.array(parents_reverse_fitness)) for item in parents_reverse_fitness]
        chrome1, chrome2 = np.random.choice(self.__list, p=parents_prob, size=2, replace=False)
        return chrome1, chrome2

    def mutation(self):
        for i in range(len(self.__list)):
            if np.random.random() > self.pm:
                continue
            swap_index = np.random.randint(2, self.n - 1)
            item = self.__list[i].get_chromosome()
            self.__list.append(Chromosome(self.n, [0]+item[swap_index:]+item[1:swap_index]))

    def select_chromosomes(self):
        parents_reverse_fitness = [1 / item.fitness(self.matrix.matrix()) for item in self.__list]
        parents_prob = [item/ sum(np.array(parents_reverse_fitness)) for item in parents_reverse_fitness]
        self.__list = list(np.random.choice(self.__list, p=parents_prob, size=self.m, replace=False))

    def mean_fitness(self):
        return np.mean([item.fitness(self.matrix.matrix()) for item in self.__list])

    def min_fitness(self):
        min = [item.fitness(self.matrix.matrix()) for item in self.__list]
        index_min = np.argmin(min)
        return self.__list[index_min], self.__list[index_min].fitness(self.matrix.matrix())
