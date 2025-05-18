import gymnasium as gym, time

def run_environment(env_name, random_actions=True, episodes=1, steps_per_episode=200, custom_action_func=None):
  env = gym.make(env_name, render_mode="human")  
  print(f"Running {env_name}") 
  print(f"Action space: {env.action_space}") 
  print(f"Observation space: {env.observation_space}") 
    
  for episode in range(episodes):
    observation, info = env.reset()
    total_reward = 0
        
    for step in range(steps_per_episode):
      if random_actions:
        action = env.action_space.sample()
      else:
        action = custom_action_func(observation)

      observation, reward, terminated, truncated, info = env.step(action)
      total_reward += reward

      env.render()
      time.sleep(0.01)

      if terminated or truncated:
        break
                
    print(f"Episode {episode+1} finished with total reward: {total_reward}")
    
  env.close()
  print(f"Finished running {env_name}\n")

# A) Uruchominie środowisk z wykładu

run_environment("LunarLander-v3", random_actions=True, episodes=2)
run_environment("FrozenLake-v1", random_actions=True, episodes=2)

# B) Uruchomienie dodatkowych gier

run_environment("CartPole-v1", random_actions=True, episodes=2)
run_environment("BipedalWalker-v3", random_actions=True, episodes=2)
run_environment("Blackjack-v1", random_actions=True, episodes=2)
# run_environment("Ant-v4", random_actions=True, episodes=1) # doinstalować "gymnasium[mujoco]"
# run_environment("ALE/Breakout-v5", random_actions=True, episodes=1) # doinstalować "gymnasium[atari]"

# c) Identyfikacja środowisk/gier

#    C1) Stan gry i zestaw akcji są dyskretne (tzn. jest to skończony zestaw).
#        1. FrozenLake-v1 (Action space: Discrete(4); Observation space: Discrete(16))
#        2. Blackjack-v1 (Action space: Discrete(2); 
#                   Observation space: Tuple(Discrete(32), Discrete(11), Discrete(2)))

#   C2) Stan gry jest ciągły (nieskończony, liczby zmiennoprzecinkowe), ale zestaw akcji jest dyskretny
#       1. LunarLander-v3 (Action space: Discrete(4)
#          Observation space: Box([ -2.5        -2.5       -10.        -10.         -6.2831855 -10.
#          -0.         -0.       ], [ 2.5        2.5       10.        10.         6.2831855 10.
#           1.         1.       ], (8,), float32))     
#       2. CartPole-v1 (Action space: Discrete(2)
#            Observation space: Box([-4.8               -inf -0.41887903        -inf],
#                       [4.8               inf 0.41887903        inf], (4,), float32))

#   C3) Stan gry i zestaw akcji są ciągłe (nieskończony, liczby zmiennoprzecinkowe) 
#       1. BipedalWalker-v3 (Action space: Box(-1.0, 1.0, (4,), float32)
#          Observation space: Box([-3.1415927 -5.        -5.        -5.        -3.1415927 -5.
#         -3.1415927 -5.        -0.        -3.1415927 -5.        -3.1415927
#         -5.        -0.        -1.        -1.        -1.        -1.
#         -1.        -1.        -1.        -1.        -1.        -1.       ], 
#         [3.1415927 5.        5.        5.        3.1415927 5.        3.1415927
#          5.        5.        3.1415927 5.        3.1415927 5.        5.
#          1.        1.        1.        1.        1.        1.        1.
#          1.        1.        1.        1.       ], (24,), float32))

# Action space - na podstawie obserwacji model wybiera jedną z akcji, jeśli
#  - dyskretna, to wybiera jedną z akcji ze zbioru (np. lewo, prawo, góra, dół),
#  - ciągła, to wybiera jedną z akcji w double (nieskończenie wiele możliwości).
# Action space jest elementem wyjściowym, który jest przekazywany z modelu. (jego "decyzja wyboru")

# Observation space przekazuje informacje o stanie gry, które są przekazywane do modelu.
# Observation space jest elementem wejściowym, który jest przekazywany do modelu. (jego perpektywa)

# Observation czyli agent widzi stan gry, a action to decyzja agenta.

# D) Stwórz własne funkcje akcji dla 2 gier 

#   1. CartPole-v1
#   Observation: [pozycja wózka, prędkość wózka, kąt drążka, prędkość kątowa drążka]
#   Action: 0 - przesunięcie w lewo, 1 - przesunięcie w prawo

def cartpole_strategy(observation):
  if observation[2] > 0: # Jeżeli klocek przechyla się w prawo
    return 1 # Wybierz akcję 1 (przesuń w prawo)
  else:
    return 0  # Wybierz akcję 0 (przesuń w lewo)

run_environment("CartPole-v1", random_actions=False, episodes=5, custom_action_func=cartpole_strategy)
# Wyniki z własnej strategii:
# Episode 1 finished with total reward: 60.0
# Episode 2 finished with total reward: 56.0
# Episode 3 finished with total reward: 56.0
# Episode 4 finished with total reward: 45.0
# Episode 5 finished with total reward: 52.0

run_environment("CartPole-v1", random_actions=True, episodes=5)
# Wyniki z losowej strategii:
# Episode 1 finished with total reward: 25.0
# Episode 2 finished with total reward: 17.0
# Episode 3 finished with total reward: 31.0
# Episode 4 finished with total reward: 12.0
# Episode 5 finished with total reward: 21.0 

#  2. Blackjack-v1
#   Observation: [suma kart gracza, karta krupiera, czy gracz ma as]
#   Action: 0 - pas, 1 - dobierz kartę
def blackjack_strategy(observation):
  player_sum = observation[0]
  if player_sum >= 17:
    return 0 
  else:
    return 1  

run_environment("Blackjack-v1", random_actions=False, episodes=10, custom_action_func=blackjack_strategy)
# Wyniki z własnej strategii:
# Episode 1 finished with total reward: 1.0
# Episode 2 finished with total reward: 1.0
# Episode 3 finished with total reward: 1.0
# Episode 4 finished with total reward: 1.0
# Episode 5 finished with total reward: -1.0
# Episode 6 finished with total reward: 1.0
# Episode 7 finished with total reward: 1.0
# Episode 8 finished with total reward: -1.0
# Episode 9 finished with total reward: 1.0
# Episode 10 finished with total reward: 0.0

run_environment("Blackjack-v1", random_actions=True, episodes=10)
# Wyniki z losowej strategii:
# Episode 1 finished with total reward: 0.0
# Episode 2 finished with total reward: -1.0
# Episode 3 finished with total reward: 1.0
# Episode 4 finished with total reward: -1.0
# Episode 5 finished with total reward: 1.0
# Episode 6 finished with total reward: -1.0
# Episode 7 finished with total reward: -1.0
# Episode 8 finished with total reward: 1.0
# Episode 9 finished with total reward: -1.0
# Episode 10 finished with total reward: -1.0