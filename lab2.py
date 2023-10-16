#Метаэвристические алгоритмы

from pickle import bytes_types
import pygmo as pg

class Rosenbrock:
  def __init__(self):
    self.dim = 2
    self.lower_bound = -2
    self.upper_bound = 2

  def fitness(self, x):
    return [-(1 - x[0])**2 - 100*(x[1] - x[0]**2)**2]
  def get_bounds(self):
    return ([self.lower_bound] * self.dim, [self.upper_bound] * self.dim)

pr = pg.problem(Rosenbrock())
algo = pg.algorithm(pg.sga(gen = 100))
population = pg.population(pr , size = 20)
population = algo.evolve(population)

best_fitness = population.get_f()[0]
best_solution = population.get_x()[0]

print(best_fitness)
print(best_solution)

x = np.linspace(-2, 2, 100) # change
y = np.linspace(-2, 2, 100) # change
X, Y = np.meshgrid(x,y)
Z = -(1 - X)**2 - 100*(Y - X**2)**2

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot_surface(X,Y,Z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.scatter(best_solution[0], best_solution[1], best_fitness, c = 'r', s = 30)
