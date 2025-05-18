import numpy as np, time, random
import gymnasium as gym

random.seed(42)
np.random.seed(42)

# Value Itheration Algorithm optymalizuję strategię w środowiska z dyskretnymi stanami i akcjami.
# Wykorzystuje algorytm iteracji wartości do znalezienia optymalnej drogi i funkcji wartości.
# Gamma oznacza współczynnik dyskontowy, który kontroluje wpływ przyszłych nagród na bieżącą decyzję.
# Theta to próg konwergencji, który określa, kiedy algorytm powinien zakończyć iterację.

# Frozen idealnie nadaje się do tego algorytmu, ponieważ ma dyskretne stany i akcje.
# Algorytm bazuje na tablicy wartości Q, która przechowuje wartości dla każdej pary stan-akcja,
# dlatego potrzebne są dyskretne stany i akcje.
def value_iteration(env, gamma=0.99, theta=1e-8):
  V = np.zeros(env.observation_space.n)
    
  while True:
    delta = 0
    for s in range(env.observation_space.n):
      v = V[s]
      action_values = []
      for a in range(env.action_space.n):
        next_state_value = 0
        for prob, next_state, reward, done in env.unwrapped.P[s][a]:
          next_state_value += prob * (reward + gamma * V[next_state] * (not done))
        action_values.append(next_state_value)
      V[s] = max(action_values)
      delta = max(delta, abs(v - V[s]))

    if delta < theta:
      break

  policy = np.zeros(env.observation_space.n, dtype=int)
  for s in range(env.observation_space.n):
    action_values = []
    for a in range(env.action_space.n):
      next_state_value = 0
      for prob, next_state, reward, done in env.unwrapped.P[s][a]:
        next_state_value += prob * (reward + gamma * V[next_state] * (not done))
      action_values.append(next_state_value)
    policy[s] = np.argmax(action_values)
    
  return V, policy

def test_policy(env, policy, delay=0.5):
  state, _ = env.reset()
  done = False
  total_reward = 0
  steps = 0
  while not done:
    action = policy[state]
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward
    steps += 1
    time.sleep(delay)  
        
  return total_reward, steps

def main():
  print("Value Iteration Algorithm (Dynamic Programming)")
  print("=" * 50)

  # Frozen Lake 4x4  
  env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False)
  V, policy = value_iteration(env, gamma=0.99)
    
  print("\nOptimal Value Function:")
  print(V.reshape(4, 4))
    
  print("\nOptimal Policy:")
  policy_chars = np.array(['←', '↓', '→', '↑'])[policy]
  print(policy_chars.reshape(4, 4))

  test_env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode="human")
  reward, steps = test_policy(test_env, policy)
  print(f"Test completed! Total reward: {reward}, Steps taken: {steps}")
  test_env.close()
    
  # Frozen Lake 8x8
  env_8x8 = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True)
    
  print("\nStarting Value Iteration for FrozenLake-v1 (8x8)...")
  V_8x8, policy_8x8 = value_iteration(env_8x8, gamma=0.99)
    
  print("\nOptimal Policy (8x8):")
  policy_chars_8x8 = np.array(['←', '↓', '→', '↑'])[policy_8x8]
  print(policy_chars_8x8.reshape(8, 8))

  test_env_8x8 = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode="human")
  reward_8x8, steps_8x8 = test_policy(test_env_8x8, policy_8x8)
  print(f"Test completed! Total reward: {reward_8x8}, Steps taken: {steps_8x8}")
  test_env_8x8.close()

main()

''' Wyniki:
Value Iteration Algorithm (Dynamic Programming)
==================================================

Optimal Value Function:
[[0.95099005 0.96059601 0.970299   0.96059601]
 [0.96059601 0.         0.9801     0.        ]
 [0.970299   0.9801     0.99       0.        ]
 [0.         0.99       1.         0.        ]]

Optimal Policy:
[['↓' '→' '↓' '←']
 ['↓' '←' '↓' '←']
 ['→' '↓' '↓' '←']
 ['←' '→' '→' '←']]
Test completed! Total reward: 1.0, Steps taken: 6

Now let's try with a larger environment (8x8) with slippery ice...

Starting Value Iteration for FrozenLake-v1 (8x8)...

Optimal Policy (8x8):
[['↑' '→' '→' '→' '→' '→' '→' '→']
 ['↑' '↑' '↑' '↑' '↑' '→' '→' '↓']
 ['↑' '↑' '←' '←' '→' '↑' '→' '↓']
 ['↑' '↑' '↑' '↓' '←' '←' '→' '→']
 ['←' '↑' '←' '←' '→' '↓' '↑' '→']
 ['←' '←' '←' '↓' '↑' '←' '←' '→']
 ['←' '←' '↓' '←' '←' '←' '←' '→']
 ['←' '↓' '←' '←' '↓' '→' '↓' '←']]
Test completed! Total reward: 0.0, Steps taken: 100'''