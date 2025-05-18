import gymnasium as gym, numpy as np, random, time 
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Deep Q-Network Algorithm działa na środowiskach z ciągłymi stanami i dyskretnymi akcjami.
# Wykorzystuje sieć neuronową do aproksymacji funkcji Q, co pozwala na działanie w przestrzeniach
# o dużej liczbie stanów, gdzie tradycyjne metody tabelaryczne (jak Q-Learning) nie są praktyczne.
# W przeciwieństwie do Value Iteration i Q-Learning, które wymagają dyskretnych stanów,
# DQN radzi sobie z ciągłymi przestrzeniami stanów.

# Algorytm wykorzystuje:
# - Experience replay: przechowywanie i ponowne wykorzystanie wcześniejszych doświadczeń
# - Target network: osobna sieć do stabilizacji uczenia
# - Epsilon-greedy policy: balans między eksploracją a eksploatacją

class DQNAgent:
  def __init__(self, state_size, action_size):
    self.state_size = state_size
    self.action_size = action_size

    # Hiperparametry
    self.memory = deque(maxlen=2000)  # Bufor experience replay
    self.gamma = 0.95                 # Współczynnik dyskontowy
    self.epsilon = 1.0                # Współczynnik eksploracji
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995
    self.learning_rate = 0.001
    self.batch_size = 64
    self.update_target_frequency = 10  # Częstotliwość aktualizacji sieci target

    self.model = self._build_model()         # Główna sieć
    self.target_model = self._build_model()  # Sieć target
    self.update_target_model()

  def _build_model(self):
    model = Sequential()
    model.add(Dense(24, input_dim=self.state_size, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(self.action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
    return model

  def update_target_model(self):
    self.target_model.set_weights(self.model.get_weights())

  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))

  def act(self, state, training=True):
    # Wybór akcji zgodnie z polityką epsilon-greedy
    if training and np.random.rand() <= self.epsilon:
      return random.randrange(self.action_size)
    
    act_values = self.model.predict(state, verbose=0)
    return np.argmax(act_values[0])

  def replay(self):
    # Uczenie na podstawie próbki doświadczeń z pamięci
    if len(self.memory) < self.batch_size:
      return
    
    minibatch = random.sample(self.memory, self.batch_size)
    
    states = np.zeros((self.batch_size, self.state_size))
    next_states = np.zeros((self.batch_size, self.state_size))
    
    for i, (state, action, reward, next_state, done) in enumerate(minibatch):
      states[i] = state
      next_states[i] = next_state
    
    # Predykcja wartości Q dla obecnych stanów przy użyciu głównej sieci
    targets = self.model.predict(states, verbose=0)
    # Predykcja wartości Q dla następnych stanów przy użyciu sieci target
    next_state_targets = self.target_model.predict(next_states, verbose=0)
    
    for i, (state, action, reward, next_state, done) in enumerate(minibatch):
      if done:
        targets[i, action] = reward
      else:
        # Double DQN: wybór akcji przez główną sieć, wartość Q z sieci target
        targets[i, action] = reward + self.gamma * np.max(next_state_targets[i])
        
    # Uczenie sieci
    self.model.fit(states, targets, epochs=1, verbose=0)
    
    # Zmniejszanie epsilon (mniej eksploracji z czasem)
    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay

def train_dqn(env, num_episodes=500, batch_size=64):
  state_size = env.observation_space.shape[0]
  action_size = env.action_space.n
  agent = DQNAgent(state_size, action_size)
  agent.batch_size = batch_size
  
  stats = {
    'episode_rewards': np.zeros(num_episodes),
    'episode_lengths': np.zeros(num_episodes)
  }
  
  for episode in tqdm(range(num_episodes), desc="Training DQN"):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    episode_reward = 0
    step_count = 0
    
    while not done:
      # Wybór akcji
      action = agent.act(state)
      # Wykonanie akcji (otrzymanie nagrody i nowego stanu)
      next_state, reward, terminated, truncated, _ = env.step(action)
      done = terminated or truncated
      next_state = np.reshape(next_state, [1, state_size])
      agent.remember(state, action, reward, next_state, done)

      state = next_state
      episode_reward += reward
      step_count += 1
      agent.replay() # Uczenie z dowiadczeń
      
      if done:
        break

    if episode % agent.update_target_frequency == 0:
      agent.update_target_model()
      

    stats['episode_rewards'][episode] = episode_reward
    stats['episode_lengths'][episode] = step_count
    if (episode + 1) % 100 == 0:
      avg_reward = np.mean(stats['episode_rewards'][max(0, episode-99):episode+1])
      print(f"Episode {episode+1}/{num_episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
  
  return agent, stats

def plot_training_stats(stats):
  plt.figure(figsize=(12, 5))
  
  plt.subplot(1, 2, 1)
  plt.plot(stats['episode_rewards'])
  plt.title('DQN: Episode Rewards')
  plt.xlabel('Episode')
  plt.ylabel('Reward')
  window_size = 100
  if len(stats['episode_rewards']) > window_size:
    smoothed_rewards = np.convolve(stats['episode_rewards'], 
                     np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, len(stats['episode_rewards'])), 
                     smoothed_rewards, 'r-', linewidth=2)
  
  plt.subplot(1, 2, 2)
  plt.plot(stats['episode_lengths'])
  plt.title('DQN: Episode Lengths')
  plt.xlabel('Episode')
  plt.ylabel('Steps')
  if len(stats['episode_lengths']) > window_size:
    smoothed_lengths = np.convolve(stats['episode_lengths'], 
                     np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, len(stats['episode_lengths'])), 
                     smoothed_lengths, 'r-', linewidth=2)
  
  plt.tight_layout()
  plt.savefig('dqn_training.png')

def test_policy(env, agent, max_steps=500, delay=0.03):
  state, _ = env.reset()
  state = np.reshape(state, [1, agent.state_size])
  done = False
  total_reward = 0
  steps = 0
  
  while not done and steps < max_steps:
    action = agent.act(state, training=False) 
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    next_state = np.reshape(next_state, [1, agent.state_size])
    state = next_state
    total_reward += reward
    steps += 1
    time.sleep(delay)  
    
  return total_reward, steps

def main():
  print("Deep Q-Network Algorithm (Reinforcement Learning)")
  print("=" * 50)

  # CartPole-v1 - środowisko z ciągłymi stanami i dyskretnymi akcjami
  env_name = 'CartPole-v1'
  env = gym.make(env_name)
  
  print(f"\nStarting DQN for {env_name}...")
  print(f"State space: {env.observation_space}")  # Ciągła przestrzeń stanów
  print(f"Action space: {env.action_space}")      # Dyskretna przestrzeń akcji (0 lub 1)
  
  # Trening agenta DQN
  agent, stats = train_dqn(env, num_episodes=500, batch_size=64)
  plot_training_stats(stats)
  
  test_env = gym.make(env_name, render_mode="human")
  reward, steps = test_policy(test_env, agent)
  print(f"Test completed! Total reward: {reward}, Steps taken: {steps}")
  test_env.close()

main()

'''Wyniki:
'''