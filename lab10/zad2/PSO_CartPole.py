import gymnasium as gym, numpy as np, time
from pyswarm import pso  

env = gym.make('CartPole-v1', render_mode="rgb_array")

n_observations = 4  # CartPole ma 4 obserwacje (położenie, prędkość, kąt, prędkość kątowa)
n_actions = 2       # i dwie akcje (lewo, prawo)
max_steps = 500
input_size = n_observations
hidden_size = 8 # Liczba neuronów w warstwie ukrytej
output_size = 1 # Liczba neuronów w warstwie wyjściowej (1 neuron dla akcji 0 lub 1 [lewo/prawo])
n_weights = (input_size * hidden_size) + hidden_size + (hidden_size * output_size) + output_size

def get_network_weights(weights):
  w1_size = input_size * hidden_size
  w1 = weights[:w1_size].reshape(input_size, hidden_size)

  b1_size = hidden_size
  b1 = weights[w1_size:w1_size + b1_size]

  w2_size = hidden_size * output_size
  start_idx = w1_size + b1_size
  w2 = weights[start_idx:start_idx + w2_size].reshape(hidden_size, output_size)

  b2 = weights[start_idx + w2_size:]
    
  return w1, b1, w2, b2

def compute_action(observation, weights):
  w1, b1, w2, b2 = get_network_weights(weights)
  hidden = np.maximum(0, np.dot(observation, w1) + b1)
  output = np.dot(hidden, w2) + b2
    
  prob = 1 / (1 + np.exp(-output))  # Sigmoid
  action = 1 if prob > 0.5 else 0    
  return action

def objective_function(weights):
    env.reset(seed=42) 
    total_reward = 0
    observation, _ = env.reset()
    done = truncated = False
    
    for _ in range(max_steps):
      if done or truncated:
        break

      action = compute_action(observation, weights)
      observation, reward, done, truncated, info = env.step(action)
      total_reward += reward
    
    # Pobieramy feedback z nagrody ze środowiska
    # Minimalizujemy funkcję celu, więc zwracamy -total_reward 
    return -total_reward

# Granice dla wag
lb = -1.0 * np.ones(n_weights)  # Dolna granica
ub = 1.0 * np.ones(n_weights)   # Górna granica

# PSO 
swarmsize = 50
maxiter = 30
print(f"Starting PSO optimization with {swarmsize} particles and {maxiter} iterations")
print(f"Each particle has {n_weights} dimensions (weights)")

best_weights, best_fitness = pso(objective_function, lb, ub, 
                                swarmsize=swarmsize, maxiter=maxiter,
                                debug=True, phip=1.5, phig=1.5)

# Sieć składa się z:
#   - Warstwa wejściowa: 4 neurony (zmienne stanu)
#   - Warstwa ukryta: 8 neuronów z aktywacją ReLU
#   - Warstwa wyjściowa: 1 neuron z aktywacją sigmoid dla wyboru akcji binarnej
# Funkcja dopasowania: Negatywna suma nagród z jednego epizodu (PSO minimalizuje)
# PSO optymalizuje wagi, aby zmaksymalizować oczekiwaną nagrodę

print(f"Optimization complete, best fitness: {-best_fitness}")

def test_best_solution(weights):
  env = gym.make('CartPole-v1', render_mode="human")
  observation, _ = env.reset(seed=42)  
  total_reward = 0
  done = truncated = False
  print("Testing best solution...")
    
  for _ in range(max_steps):
    if done or truncated:
      break
        
    action = compute_action(observation, weights)
    observation, reward, done, truncated, info = env.step(action)
    total_reward += reward
    time.sleep(0.01)  
    
  print(f"Testing completed. Total reward: {total_reward}")
  env.close()
  return total_reward

test_best_solution(best_weights)