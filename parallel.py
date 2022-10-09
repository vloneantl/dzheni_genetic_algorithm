import numpy as np

from population import Population
from Matrix import Matrix
import json
from joblib import Parallel, delayed
# np.random.seed(100)

with open('config.cfg', 'r') as infile:
    config = json.load(infile)
    print(config)

m = config['m'] #количество особей в популяции
n = config['n'] #количество городов
k = config['k'] #количество родителей
z = config['z'] #точность остановки алгоритма
l = config['l'] #количество эпох
pc = config['pc'] #вероятность кроссовера
pm = config['pm'] #вероятность мутации
min_dist = config['min_dist'] # мин расстояние между городами
max_dist = config['max_dist'] # макс расстояние между городами

matr1 = Matrix(n=n, min_dist=min_dist, max_dist=max_dist)
matr1.generate()
matr1.save_csv()
matr1.read_csv()
matr1.print_array()
populations = [Population(matr1, int(m/10), n, pc, pm) for _ in range(10)]

def one_algorithm(pop):
    epochs = 0
    prev_mean_fitness = 0
    medians = []
    best = np.Inf
    while epochs < l and abs(pop.mean_fitness() - prev_mean_fitness) >= z:
        prev_mean_fitness = pop.mean_fitness()
        for _ in range(k):
            choice = pop.parents_selection()
            pop.crossover(*choice)
        pop.mutation()
        epochs += 1
        pop.select_chromosomes()
        medians.append(pop.mean_fitness())
        if pop.min_fitness()[1] < best:
            best_chrome, best = pop.min_fitness()
    return best_chrome.get_chromosome(), best


a = Parallel(n_jobs=-1)(delayed(one_algorithm)(populations[i]) for i in range(10))
print(a, sep='\n')