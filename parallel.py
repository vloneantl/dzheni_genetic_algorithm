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
min_dist = config['min_dist'] #мин расстояние между городами
max_dist = config['max_dist'] #макс расстояние между городами
pop_cnt = config['pop_cnt'] #количество подпопуляций
epoch_before_mix = config['epoch_before_mix'] #эпохи до смешивания


matr1 = Matrix(n=n, min_dist=min_dist, max_dist=max_dist)
matr1.generate()
matr1.save_csv()
matr1.read_csv()
matr1.print_array()
populations = [Population(matr1, int(m/pop_cnt), n, pc, pm) for _ in range(pop_cnt)]

def one_algorithm(pop):
    epochs = 0
    prev_mean_fitness = 0
    medians = []
    best = np.Inf
    while epochs < epoch_before_mix and abs(pop.mean_fitness() - prev_mean_fitness) >= z:
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

def swap(population1, population2):
    arr = population1.get_list() + population2.get_list()
    np.random.shuffle(arr)
    population1.set_list(arr[:len(arr)//2]), population2.set_list(arr[len(arr)//2:])
    return population1, population2

def mixed():
    result = Parallel(n_jobs=-1)(delayed(one_algorithm)(populations[i]) for i in range(pop_cnt))
    for _ in range(5):
        a = np.random.choice(list(range(pop_cnt)), size=2, replace=False)
        populations[a[0]], populations[a[1]] = swap(populations[a[0]], populations[a[1]])
    return result

def final():
    for i in range(300):
        result = mixed()
    return result

print(*final(), sep='\n')