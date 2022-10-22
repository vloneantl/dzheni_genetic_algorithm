import numpy as np

from population import Population
from Matrix import Matrix
import matplotlib.pyplot as plt
import json
np.random.seed(1337)

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

matr1 = Matrix(n=n, min_dist=min_dist, max_dist=max_dist, trivial_definition=1)
# matr1.generate()
# matr1.save_csv()
matr1.read_csv()
matr1.print_array()

pop = Population(matr1, m, n, pc, pm)
epochs = 0
prev_mean_fitness = 0
medians = []
best = np.Inf
while epochs < l:# and abs(pop.mean_fitness()-prev_mean_fitness) >= z:
    print('epoch',epochs)
    prev_mean_fitness = pop.mean_fitness()
    for _ in range(k):
        choice = pop.parents_selection()
        pop.crossover(*choice)
    pop.mutation()
    epochs+=1
    pop.select_chromosomes()
    print(pop.mean_fitness())
    medians.append(pop.mean_fitness())
    if pop.min_fitness()[1] < best:
        best_chrome, best = pop.min_fitness()

print('finished, done', epochs,'epochs')
print('best_fitness:',best, best_chrome.get_chromosome())
plt.plot(medians)
plt.show()
