import gymnasium as gym, numpy as np, time, random
import matplotlib.pyplot as plt
from tqdm import tqdm

random.seed(42)
np.random.seed(42)

# Q-Learning Algorithm działa na środowiskach z dyskretnymi stanami i akcjami. Działa na zasadzie 
# aktualizacji Q-wartości dla każdej pary stan-akcja na podstawie nagród i wartości Q następnego stanu.
# Czyli uczy się z doświadczenia, oznacza to, że każda akcja wpływa na przyszłe decyzje.
# Im więcej razy algorytm będzie się uczył, tym będzie lepszy.

# num_episodes: liczba epizodów do nauki
# alpha: współczynnik uczenia (learning rate)
# gamma: współczynnik dyskontowy (discount factor), czyli jak bardzo przyszłe nagrody są ważne
# epsilon: współczynnik eksploracji (exploration rate)

# Taxi-v3 jest środowiskiem z dyskretnymi stanami i akcjami, które jest idealne do nauki Q-Learning.
# Algorytm bazuje na tablicy wartości Q, która przechowuje wartości dla każdej pary stan-akcja,
# dlatego potrzebne są dyskretne stany i akcje.
def q_learning(env, num_episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1):
	Q = np.zeros((env.observation_space.n, env.action_space.n))
	stats = {
			'episode_rewards': np.zeros(num_episodes),
			'episode_lengths': np.zeros(num_episodes)
	}
		
	epsilon_start = epsilon
	epsilon_min = 0.01
	epsilon_decay = 0.995
		
	for episode in tqdm(range(num_episodes), desc="Training Q-Learning"):
		state, _ = env.reset()
		done = False
		episode_reward = 0
		step_count = 0
				
		while not done:
			if np.random.random() < epsilon:
				action = env.action_space.sample() 
			else:
				action = np.argmax(Q[state])  
						
			# Wykonanie akcji (otrzymanie nagrody i nowego stanu)
			next_state, reward, terminated, truncated, _ = env.step(action)
			done = terminated or truncated
			# Aktualizacja Q-wartości
			best_next_action = np.argmax(Q[next_state])
			td_target = reward + gamma * Q[next_state, best_next_action] * (not done)
			td_error = td_target - Q[state, action]
			Q[state, action] += alpha * td_error
			# Aktualizacja statystyk
			episode_reward += reward
			step_count += 1
			state = next_state
				
		stats['episode_rewards'][episode] = episode_reward
		stats['episode_lengths'][episode] = step_count

		epsilon = max(epsilon_min, epsilon * epsilon_decay)
		if (episode + 1) % 500 == 0:
			avg_reward = np.mean(stats['episode_rewards'][max(0, episode-499):episode+1])
			print(f"Episode {episode+1}/{num_episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.4f}")
		
	return Q, stats

def plot_training_stats(stats):
	plt.figure(figsize=(12, 5))
		
	plt.subplot(1, 2, 1)
	plt.plot(stats['episode_rewards'])
	plt.title('Q-Learning: Episode Rewards')
	plt.xlabel('Episode')
	plt.ylabel('Reward')
	window_size = 100
	smoothed_rewards = np.convolve(stats['episode_rewards'], 
					 np.ones(window_size)/window_size, mode='valid')
	plt.plot(range(window_size-1, len(stats['episode_rewards'])), 
					 smoothed_rewards, 'r-', linewidth=2)
		
	plt.subplot(1, 2, 2)
	plt.plot(stats['episode_lengths'])
	plt.title('Q-Learning: Episode Lengths')
	plt.xlabel('Episode')
	plt.ylabel('Steps')
	smoothed_lengths = np.convolve(stats['episode_lengths'], 
					 np.ones(window_size)/window_size, mode='valid')
	plt.plot(range(window_size-1, len(stats['episode_lengths'])), 
					 smoothed_lengths, 'r-', linewidth=2)
	
	plt.tight_layout()
	plt.savefig('q_learning_training.png')

def test_policy(env, policy, max_steps=100, delay=0.5):
	state, _ = env.reset()
	done = False
	total_reward = 0
	steps = 0
		
	while not done and steps < max_steps:
		action = policy[state]
		state, reward, terminated, truncated, _ = env.step(action)
		done = terminated or truncated
		total_reward += reward
		steps += 1
		time.sleep(delay)  
				
	return total_reward, steps

def main():
	print("Q-Learning Algorithm (Reinforcement Learning)")
	print("=" * 50)

	# Taxi-v3	
	env_name = 'Taxi-v3'
	env = gym.make(env_name)
		
	print(f"\nStarting Q-Learning for {env_name}...")
	print(f"State space size: {env.observation_space.n}")
	print(f"Action space size: {env.action_space.n}")
		
	Q, stats = q_learning(env, num_episodes=10000, alpha=0.1, gamma=0.99, epsilon=0.1)
	plot_training_stats(stats)
	policy = np.argmax(Q, axis=1)
	num_eval_episodes = 100
	eval_rewards = []
		
	for _ in range(num_eval_episodes):
		state, _ = env.reset()
		done = False
		episode_reward = 0
				
		while not done:
			action = policy[state]
			state, reward, terminated, truncated, _ = env.step(action)
			done = terminated or truncated
			episode_reward += reward
						
		eval_rewards.append(episode_reward)
		
	print(f"Average reward over {num_eval_episodes} episodes: {np.mean(eval_rewards):.2f}")

	test_env = gym.make(env_name, render_mode="human")
	reward, steps = test_policy(test_env, policy)
	print(f"Total reward: {reward}, Steps taken: {steps}")
	test_env.close()

	# FrozenLake-v1 (8x8)	
	env2 = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False)
		
	print(f"\nStarting Q-Learning for FrozenLake-v1 (8x8)...")
	print(f"State space size: {env2.observation_space.n}")
	print(f"Action space size: {env2.action_space.n}")
		
	Q2, stats2 = q_learning(env2, num_episodes=15000, alpha=0.1, gamma=0.99, epsilon=0.2)
	policy2 = np.argmax(Q2, axis=1)

	test_env2 = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode="human")
	reward2, steps2 = test_policy(test_env2, policy2)
	print(f"Total reward: {reward2}, Steps taken: {steps2}")
	test_env2.close()

main()

''' Wyniki:
Q-Learning Algorithm (Reinforcement Learning)
==================================================

Starting Q-Learning for Taxi-v3...
State space size: 500
Action space size: 6
Training Q-Learning:   5%|███▏                                                                  | 462/10000 [00:02<00:32, 291.89it/s] 
{...}
Training Q-Learning: 100%|███████████████████████████████████████████████████████████████████| 10000/10000 [00:07<00:00, 1363.48it/s]
Average reward over 100 episodes: 8.04
Total reward: 9, Steps taken: 12

Starting Q-Learning for FrozenLake-v1 (8x8)...
State space size: 64
Action space size: 4
Training Q-Learning:   3%|██▏                                                                   | 481/15000 [00:01<00:40, 358.35it/s]
{...}
Training Q-Learning: 100%|████████████████████████████████████████████████████████████████████| 15000/15000 [00:40<00:00, 373.74it/s]
Total reward: 0.0, Steps taken: 100
'''
# Taxi-v3 wykazuje lepsze wyniki niż FrozenLake-v1 (8x8)
# Taxi-v3, algorytm Q-Learning osiągnął średnią nagrodę 8.04 w 100 epizodach testowych, wynik dobry
# FrozenLake-v1 (8x8), algorytm nie był w stanie przejść do celu w 100 krokach, co sugeruje, że 
# potrzebne jest dalsze dostrajanie/nauka algorytmu.