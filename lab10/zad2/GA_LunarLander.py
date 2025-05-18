import gymnasium as gym, pygad, time
import matplotlib.pyplot as plt

env = gym.make('LunarLander-v3', render_mode="rgb_array")

max_steps = 300  
num_solutions = 40  
num_genes = max_steps 
num_generations = 50
num_parents_mating = 15
mutation_probability = 0.05

def fitness_func(ga_instance, solution, solution_idx):
  env.reset(seed=42)
  total_reward = 0
  done = truncated = False

  for action in solution:
    if done or truncated:
      break

    action = int(action)
    observation, reward, done, truncated, info = env.step(action)
    total_reward += reward

    # Użycie nagród wbudowanych w środowisko LunarLander
    # Nagrody za lądowanie, karanie za nieudane lądowanie
  return total_reward

ga_instance = pygad.GA(
  num_generations=num_generations,
  num_parents_mating=num_parents_mating,
  fitness_func=fitness_func,
  sol_per_pop=num_solutions,
  num_genes=num_genes,
  gene_type=float,
  gene_space=[0, 1, 2, 3],  # Action space: None, Left, Main, Right engines
  init_range_low=0,
  init_range_high=4,
  parent_selection_type="rank",
  crossover_type="two_points",
  crossover_probability=0.9,
  mutation_type="random",
  mutation_probability=mutation_probability,
  mutation_by_replacement=True,
  keep_elitism=2,
  save_best_solutions=True,
  random_seed=42
)

# Każdy chromosom to lista akcji (0-3) reprezentujących ruchy (None, Left, Main, Right)
# przez maksymalnie max_steps kroków. Reprezentacja genów jako liczby zmiennoprzecinkowe.
# Algorytm genetyczny ewoluuje populację przez pokolenia, wybierając rodziców na podstawie
# selekcji rankingowej.

ga_instance.run()
solution, solution_fitness, _ = ga_instance.best_solution()
print(f"Best fitness: {solution_fitness}")
print(f"Best solution (first 20 actions): {solution[:20]}...")

def test_solution(solution):
  env = gym.make('LunarLander-v3', render_mode="human")
  obs, _ = env.reset(seed=42)
  total_reward = 0
  done = truncated = False
    
  print("Testing best solution...")
  for action in solution:
    if done or truncated:
      break
        
    action = int(action)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    time.sleep(0.02)  
    
  print(f"Testing completed. Total reward: {total_reward}")
  env.close()
  return total_reward

plt.figure(figsize=(10, 6))
plt.plot(ga_instance.best_solutions_fitness)
plt.title('Fitness over Generations for LunarLander')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.grid(True)
plt.savefig('Plot_LunarLander_Fitness.png')

test_solution(solution)