import pygad
import time
import numpy as np
import matplotlib.pyplot as plt

# 1 - walls, 0 - path
maze = np.array([
  [1,1,1,1,1,1,1,1,1,1,1,1],
  [1,0,0,0,1,0,0,0,1,0,0,1],
  [1,1,1,0,0,0,1,0,1,1,0,1],
  [1,0,0,0,1,0,1,0,0,0,0,1],
  [1,0,1,0,1,1,0,0,1,1,0,1],
  [1,0,0,1,1,0,0,0,1,0,0,1],
  [1,0,0,0,0,0,1,0,0,0,1,1],
  [1,0,1,0,0,1,1,0,1,0,0,1],
  [1,0,1,1,1,0,0,0,1,1,0,1],
  [1,0,1,0,1,1,0,1,0,1,0,1],
  [1,0,1,0,0,0,0,0,0,0,0,1],
  [1,1,1,1,1,1,1,1,1,1,1,1]
])

start = (1, 1)
end = (10, 10)

possible_moves = [(0, 1), (1, 0), (0, -1), (-1, 0)] 
max_steps = 30

sol_per_pop = 75
num_genes = max_steps
num_parents_mating = 25
num_generations = 150
keep_parents = 10
parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 4 

def is_valid_move(maze, position, move):
  new_position = (position[0] + move[0], position[1] + move[1])
  if (new_position[0] < 0 or new_position[0] >= maze.shape[0] or
      new_position[1] < 0 or new_position[1] >= maze.shape[1]):
    return False
  if maze[new_position] == 1:
    return False
  return True

gene_space = [0,1,2,3] # 0 - right, 1 - down, 2 - left, 3 - up 

def fitness_func(ga_instance, solution, solution_idx):
  current_position = start
  steps = 0
  path_length = 0 

  for move_idx in solution:
    if steps >= max_steps:
      break
    move = possible_moves[int(move_idx)]
    steps += 1
    if is_valid_move(maze, current_position, move):
      current_position = (current_position[0] + move[0], current_position[1] + move[1])
      path_length += 1
      if current_position == end:
        return steps*(-1)
    else:
      steps +=1000
    
  distance_to_end = abs(current_position[0]-end[0]) + abs(current_position[1]-end[1])
  return (1000+max_steps+distance_to_end)*-1

def on_generation(ga_instance):
  solution, fitness, _ = ga_instance.best_solution()
  if fitness <= -1:
    return "reached_end"
  
def run_ga():
  start_time = time.time()
  ga_instance=pygad.GA(gene_space=gene_space,
              num_generations=num_generations,
              num_parents_mating=num_parents_mating,
              fitness_func=fitness_func,
              sol_per_pop=sol_per_pop,
              num_genes=num_genes,
              parent_selection_type=parent_selection_type,
              keep_parents=keep_parents,
              crossover_type=crossover_type,
              mutation_type=mutation_type,
              mutation_percent_genes=mutation_percent_genes,
  )

  ga_instance.run()
  end_time = time.time()
  execution_time = end_time-start_time

  solution, solution_fitness, solution_idx = ga_instance.best_solution()

  solution_found = solution_fitness >= -30 
  return solution, solution_fitness, execution_time, solution_found, ga_instance

def test_ga(num_runs=10):
  succes_count = 0
  succes_times = []
  best_solution = None
  best_fitness = float('inf')
  best_ga_instance = None

  for i in range(num_runs):
    print(f'Run {i+1}/{num_runs}')
    solution, fitness, exec_time, solution_found, ga_instance = run_ga()

    if solution_found:
      succes_count +=1
      succes_times.append(exec_time)
      if fitness < best_fitness:
        best_solution = solution
        best_fitness = fitness
        best_ga_instance = ga_instance
    elif best_solution is None:
      best_solution = solution
      best_fitness = fitness
      best_ga_instance = ga_instance

  sucess_rate = (succes_count / num_runs) * 100
  avg_time = np.mean(succes_times) if succes_times else 0
  print(f'Results from {num_runs} runs:')
  print(f'Success rate: {sucess_rate}%')
  print(f'Average time for successful runs: {avg_time} seconds')

  return best_solution, best_fitness, best_ga_instance

def decode_solution(solution):
  current_position = start
  path = [current_position]
  steps=0

  for move_idx in solution:
    if steps >= max_steps:
      break
    move = possible_moves[int(move_idx)]
    if is_valid_move(maze, current_position, move):
      current_position = (current_position[0] + move[0], current_position[1] + move[1])
      path.append(current_position)
      steps += 1
      if current_position == end:
        break
  return path

def display_path(maze, path):
  maze_with_path = maze.copy()
  for position in path:
    maze_with_path[position] = 2

  plt.imshow(maze_with_path, cmap='Pastel2', interpolation='nearest')
  plt.colorbar()
  plt.title("Maze with Path")
  plt.savefig("maze_with_path.png")

def main():
  print("Maze navigation")

  print("\nSingle run:")
  solution, fitness, execution_time, solution_found, ga_instance=run_ga()
  print(f'Best solution fitness: {fitness}') # -20 
  print(f'Best solution: {solution}')
  print(f'Solution found: {solution_found}') # True 
  print(f'Execution time: {execution_time}') # 0.263s
  
  path = decode_solution(solution)
  print(f'Path taken: {path}') 
  #[(1, 1), (1, 2), (1, 3), (2, 3), (2, 4), (2, 5), (1, 5), (1, 6), (1, 7), (2, 7), (3, 7),
  #    (4, 7), (5, 7), (6, 7), (6, 8), (6, 9), (7, 9), (7, 10), (8, 10), (9, 10), (10, 10)]
  print(f'Path length: {len(path)}') # 21
  display_path(maze, path)
  ga_instance.plot_fitness().savefig("maze_fitness_evolution.png")
  
  print("-"*50)
  print("\nTesting on 10 runs:")
  best_solution, best_fitness, best_ga_instance = test_ga(num_runs=10)
  print(f'Best solution fitness: {best_fitness}') # -26
  print(f'Best solution: {best_solution}') 
  # [0. 2. 0. 0. 1. 0. 2. 0. 0. 2. 0. 3. 0. 0. 1. 1. 1. 1. 1. 0. 0. 1. 0. 1. 1. 1. 0. 1. 1. 0.]
  # Time - 0.256s
  best_ga_instance.plot_fitness().savefig("maze_fitness_evolution_10_runs.png")

main()