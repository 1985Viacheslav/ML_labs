# Реализация классического град спуск
import numpy as np

def compute_gradient(func, x):
  epsilon = 1e-6
  gradient = np.zeros_like(x)

  for i in range(len(x)):
    x_plus = np.copy(x)
    x_plus[i] += epsilon
    fx_plus = func(x_plus)

    x_minus = np.copy(x)
    x_minus[i] -= epsilon
    fx_minus = func(x_minus)

    gradient[i] = (fx_plus - fx_minus) / (2 * epsilon)
  return gradient.astype(np.float64)

def gradiend_descent(func, initial_point, learning_rate, num_itterations):
  point = initial_point
  for _ in range(num_itterations):
    gradient = compute_gradient(func, point)
    point = point - learning_rate * gradient
  return point
#Пайплайн
def test_optimization_algo(func, optimisation_algo, initial_point, learning_rate, num_itterations):
  optimal_point = optimisation_algo(func, initial_point,learning_rate,num_itterations)
  error = np.abs(func(optimal_point))
  print("Точки оптимума",optimal_point)
  print("Значение функ",func(optimal_point))
  print("Погрешность ",error)

def rosenbrock(x):
  return -(1 - x[0])**2 - 100*(x[1] - x[0]**2)**2 #можно поменять 100 - это параметр

def sphere(x):
  return np.power(x[0],2) + np.power(x[1],2)
import matplotlib.pyplot as plt

def visualize_func(func,func_name):
  x = np.linspace(-2, 2, 100) # change
  y = np.linspace(-2, 2, 100) # change
  X, Y = np.meshgrid(x,y)
  Z = func([X,Y])

  fig = plt.figure()
  ax = fig.add_subplot(111, projection = '3d')
  ax.plot_surface(X,Y,Z)
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')

  ax.set_title(func_name)
  plt.show()

visualize_func(rosenbrock, 'rosenbrock')
visualize_func(sphere, 'sphere')

def gradiend_descent(func, initial_point, learning_rate, num_itterations):
  point = initial_point
  trajectory = [point]
  for _ in range(num_itterations):
    gradient = compute_gradient(func, point)
    point = point - learning_rate * gradient
    trajectory.append(point)
  return point, trajectory


def visualize_optimization(func,optimization_algo, initial_point, learning_rate, num_itterations,func_name):
  x = np.linspace(-2, 2, 100) # change
  y = np.linspace(-2, 2, 100) # change
  X, Y = np.meshgrid(x,y)
  Z = func([X,Y])
  fig = plt.figure()
  ax = fig.add_subplot(111, projection = '3d')
  optimal_point, trajectory = optimization_algo(func,initial_point, learning_rate, num_itterations)
  trajectory = np.array(trajectory)

  ax.scatter(trajectory[:, 0], trajectory[:, 1], func(trajectory.T), color = 'r', s = 50)
  ax.scatter(optimal_point[0], optimal_point[1], func(optimal_point), color = 'g', s = 100)
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')

  ax.set_title(func_name)

  plt.show()


initial_point = np.array([1,1]) # change
learning_rate = 0.00001
num_itterations = 10000

visualize_optimization(rosenbrock, gradiend_descent, initial_point, learning_rate, num_itterations,'rosenbrock')

initial_point = np.array([1,1]) # change
learning_rate = 0.001
num_itterations = 100000

visualize_optimization(sphere, gradiend_descent, initial_point, learning_rate, num_itterations,'sphere')



initial_point = np.array([1,1]) # change
learning_rate = 0.0001
num_itterations = 10000

test_optimization_algo(rosenbrock, gradiend_descent, initial_point, learning_rate, num_itterations)
test_optimization_algo(sphere, gradiend_descent, initial_point, learning_rate, num_itterations)

# Моментная
def momentum_gradient_descent(func, initial_point, learning_rate, num_itterations, momentum = 0.9):
  point = initial_point
  for _ in range(num_itterations):
    gradient = compute_gradient(func, point)
    velocity = momentum * velocity + learning_rate * gradient
    point = point - velocity
  return point


#Адаптивная
def adam_gradient_descent(func, initial_point, learning_rate, num_itterations, beta1 = 0.9, beta2 = 0.9, epsilon = 1e-8):
  point = initial_point
  m = np.zeros_like(point)
  v = np.zeros_like(point)

  for t in range(1, num_itterations + 1):
    gradient = compute_gradient(func, point)
    m = beta1 * m + (1 - beta1) * gradient
    v = beta2 * v + (1 - beta2) * gradient ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    point = point - learning_rate * m_hat/ (np.sqrt(v_hat) + epsilon)

  return point

#Эволюция темпа
import random

def evolution_learning_rate(func, initial_point, num_itterations, population_size, mutation_rate):
  population = [initial_point] * population_size
  best_point = initial_point

  for _ in range(num_itterations):
    fitness_values = [func(point) for point in population]
    best_index = fitness_values.index(min(fitness_values))
    best_point = population[best_index]

    for i in range(population_size):
      if random.random() < mutation_rate:
        population[i] = np.random.uniform(-2, -2, size = 2)
      else:
        population[i] = best_point
  return best_point

