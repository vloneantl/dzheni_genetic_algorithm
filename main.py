from population import Population
from Matrix import Matrix
import matplotlib.pyplot as plt
import json
#np.random.seed(42)

with open('config.cfg', 'r') as infile:
    config = json.load(infile)
    print(config)

m = config['m']
n = config['n']
k = config['k']
z = config['z']
l = config['l']
pc = config['pc']
pm = config['pm']
min_dist = config['min_dist']
max_dist = config['max_dist']

matr1 = Matrix(n=n, min_dist=min_dist, max_dist=max_dist)
matr1.generate()
matr1.save_csv()
matr1.read_csv()
matr1.print_array()

pop = Population(matr1, m, n, pc, pm)
epochs = 0
prev_median_fitness = 0
medians = []
while epochs < l and abs(pop.mean_fitness()-prev_median_fitness) >= z:
    print('epoch',epochs)
    prev_mean_fitness = pop.mean_fitness()
    for _ in range(k):
        choice = pop.parents_selection()
        pop.crossover(*choice)
    pop.mutation()
    epochs+=1
    pop.select_chromosomes()
    print(pop.mean_fitness())
    print(abs(pop.mean_fitness()-prev_mean_fitness))
    medians.append(pop.mean_fitness())
print('finished, done', epochs,'epochs')
print('min_fitness:',pop.min_fitness())
plt.plot(medians)
plt.show()
