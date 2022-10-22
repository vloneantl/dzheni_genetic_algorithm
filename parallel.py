import numpy as np
from threading import Thread
import matplotlib.pyplot as plt

from population import Population
from Matrix import Matrix
import json
from joblib import Parallel, delayed

# np.random.seed(100)

from multiprocessing.pool import ThreadPool
pool = ThreadPool(processes=10)

with open('config.cfg', 'r') as infile:
    config = json.load(infile)
    # print(config)

m = config['m']  # количество особей в популяции
n = config['n']  # количество городов
k = config['k']  # количество родителей
z = config['z']  # точность остановки алгоритма
l = config['l']  # количество эпох
pc = config['pc']  # вероятность кроссовера
pm = config['pm']  # вероятность мутации
min_dist = config['min_dist']  # мин расстояние между городами
max_dist = config['max_dist']  # макс расстояние между городами
pop_cnt = config['pop_cnt']  # количество подпопуляций
epoch_before_mix = config['epoch_before_mix']  # эпохи до смешивания
population_pairs_to_swap = config["population_pairs_to_swap"]  # количество пар популяций для миграции
migration_part = config['migration_part']  # часть особей, которые будут мигрировать из одной популяции в другую

matr1 = Matrix(n=n, min_dist=min_dist, max_dist=max_dist, trivial_definition=1)
matr1.generate()
matr1.save_csv()
matr1.read_csv()
# matr1.print_array()
populations = [Population(matr1, int(m / pop_cnt), n, pc, pm) for _ in range(pop_cnt)]


def train_one_population(pop):
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
    sorted_population1 = population1.sort_list().get_list()
    sorted_population2 = population2.sort_list().get_list()[::-1]
    chromosome_item_count_to_migrate = int(len(sorted_population1) * migration_part)
    sorted_population1, sorted_population2 = \
        [sorted_population2[:chromosome_item_count_to_migrate] + sorted_population1[chromosome_item_count_to_migrate:],
         sorted_population1[:chromosome_item_count_to_migrate] + sorted_population2[chromosome_item_count_to_migrate:]]

    population1.set_list(sorted_population1), population2.set_list(sorted_population2)
    return population1, population2


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        if self._target is not None:
            self._return = self._target(self._args)
    def join(self):
        Thread.join(self)
        return self._return

def train_population_and_swap():
    result= []
    threads = []
    for i in range(len(populations)):
        x = ThreadWithReturnValue(target=train_one_population, args=populations[i])
        x.start()
        threads.append(x)
    for thread in threads:
        result.append(thread.join())
    for _ in range(population_pairs_to_swap):
        population_items_to_swap = np.random.choice(list(range(pop_cnt)), size=2, replace=False)
        populations[population_items_to_swap[0]], populations[population_items_to_swap[1]] = \
            swap(populations[population_items_to_swap[0]], populations[population_items_to_swap[1]])
    result.sort(key=lambda x: x[1])
    return result[0]


def final():
    bests = []
    for i in range(l):
        result = train_population_and_swap()
        bests.append(result[1])
        print(i, result)
    plt.plot(bests)
    plt.show()
    return result


print(final())