import matplotlib.pyplot as plt
import gymnasium as gym, pygad

env = gym.make('FrozenLake8x8-v1', is_slippery=False, render_mode="rgb_array")

max_steps = 64  
num_solutions = 50 
num_genes = max_steps
num_generations = 100
num_parents_mating = 20
mutation_probability = 0.1

def fitness_func(ga_instance, solution, solution_idx):
  env.reset()
  total_reward = 0
  done = truncated = False
  steps_taken = 0

  for action in solution:
    if done or truncated:
      break
        
    action = int(action)
    observation, reward, done, truncated, info = env.step(action)
    total_reward += reward
    steps_taken += 1

    if done and reward == 1:
      total_reward += 100  # Duża nagroda za osiągnięcie celu
      total_reward += (max_steps - steps_taken) * 0.5  # Nagroda za wydajność
      break
      
  if not done or reward == 0:
    current_pos = observation
    goal_pos = 63 # Pozycja celu

    # Dystans Manhattan
    current_row, current_col = current_pos // 8, current_pos % 8
    goal_row, goal_col = goal_pos // 8, goal_pos % 8
        
    distance = abs(current_row - goal_row) + abs(current_col - goal_col)
    distance_reward = 1.0 / (distance + 1)  # Nagroda za mniejszy dystans
    total_reward += distance_reward * 10
    
  return total_reward

# Funckja fitness_func ocenia, jak dobrze sekwencja akcji działa, przyznając:
# 1. Dużą nagrodę za osiągnięcie celu
# 2. Nagrodę za szybkie osiągnięcie celu
# 3. Jeśli cel nie został osiągnięty, używa dystansu Manhattan do celu jako częściowej nagrody

ga_instance = pygad.GA(
  num_generations=num_generations,
  num_parents_mating=num_parents_mating,
  fitness_func=fitness_func,
  sol_per_pop=num_solutions,
  num_genes=num_genes,
  gene_type=float,
  gene_space=[0, 1, 2, 3],  # Action space: LEFT, DOWN, RIGHT, UP
  init_range_low=0,
  init_range_high=4,
  parent_selection_type="tournament",
  K_tournament=5,
  crossover_type="single_point",
  crossover_probability=0.9,
  mutation_type="random",
  mutation_probability=mutation_probability,
  mutation_by_replacement=True,
  keep_elitism=2,
  save_best_solutions=True
)

# Każdy chromosom to lista akcji (0-3) reprezentujących ruchy (LEFT, DOWN, RIGHT, UP) 
# przez maksymalnie max_steps kroków. Reprezentacja genów jako liczby zmiennoprzecinkowe.
# Algrytm genetyczny ewoluuje populację przez pokolenia, wybierając rodziców na podstawie
# selekcji turniejowej, przy losowej mutacji
ga_instance.run()

solution, solution_fitness, _ = ga_instance.best_solution()
print(f"Best fitness: {solution_fitness}")
print(f"Best solution: {solution[:20]}...") 

def test_solution(solution):
    env = gym.make('FrozenLake8x8-v1', is_slippery=False, render_mode="human")
    env.reset()
    total_reward = 0
    done = truncated = False
    frames = []
    
    print("Testing best solution...")
    for action in solution:
        if done or truncated:
            break
        
        action = int(action)
        observation, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        if done and reward == 1:
            print("Goal reached! Total reward:", total_reward)
            break
    
    if not (done and reward == 1):
        print("Goal not reached. Total reward:", total_reward)
    
    env.close()
    return total_reward

plt.figure(figsize=(10, 6))
plt.plot(ga_instance.best_solutions_fitness)
plt.title('Fitness over Generations')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.grid(True)
plt.savefig('Plot_Frozen_fitness.png')

test_solution(solution)