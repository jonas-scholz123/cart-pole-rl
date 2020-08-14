import gym
import numpy as np
import math

class CartPole():
    def __init__(self, max_t, n_bins, max_unbounded, discount_factor, min_epsilon,
        min_learning_rate, learning_decay, test_sample, test_sample_min_avg):
        self.max_t = max_t
        self.max_unbounded = max_unbounded
        self.discount_factor = discount_factor
        self.env = gym.make('CartPole-v0')
        self.n_bins = n_bins
        self.q_table = np.zeros([self.n_bins for i in range(self.env.observation_space.shape[0])] + [2])
        self.min_epsilon = min_epsilon
        self.min_learning_rate = min_learning_rate
        self.learning_decay = learning_decay
        self.test_sample = test_sample
        self.test_sample_min_avg = test_sample_min_avg

        self.episode_rewards = []
        self.q_sum = []

    def run_single_episode(self, render = False):
        current_observation = self.continuous_to_discrete(self.env.reset()) #initial observation
        for t in range(self.max_t):
            if render:
                self.env.render()

            action = self.choose_action(current_observation)
            new_observation, reward, done, _ = self.env.step(action)
            new_observation = self.continuous_to_discrete(new_observation)

            self.update_q(current_observation, action, reward, new_observation)

            current_observation = new_observation

            if done:
                return reward, t+1

    def run_n_episodes(self, n):
        for episode_nr in range(n):
            self.epsilon = self.get_epsilon(episode_nr)
            self.learning_rate = self.get_learning_rate(episode_nr)
            reward, termination_time = self.run_single_episode()
            self.episode_rewards.append(termination_time)
            self.q_sum.append((self.q_table).sum())

            #training completed
            if np.average(self.episode_rewards[-self.test_sample:]) > self.test_sample_min_avg:
                break

        print("FINISHED TRAINING")

        #showcase a small number of rendered simulations
        for showcase_nr in range(5):
            self.run_single_episode(render=True)

        self.env.close()

    def get_epsilon(self, t):
        return max(self.min_epsilon, np.exp(-self.learning_decay*t))

    def get_learning_rate(self, t):
        return max(self.min_learning_rate, np.exp(-self.learning_decay*t))

    def update_q(self, old_state, action, reward, new_state):
        self.q_table[old_state][action] += self.learning_rate * (reward +
            self.discount_factor * max(self.q_table[new_state])
            -self.q_table[old_state][action])

    def choose_action(self, state):
        if np.random.random() <= self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def continuous_to_discrete(self, observation):
        '''
        Turns continuous observation into discrete observation based on self.n_bins
        and self.max_unbounded, which replaces unbounded state variables.

        ======================================================
        PARAMS:
            observation : array_like
                Observation array of continous values returned by gym env
        '''
        discrete_observation = np.zeros(observation.shape)
        maximum_values = self.env.observation_space.high
        infinity_replacements = np.zeros(observation.shape)
        maximum_values = np.where(maximum_values > self.max_unbounded, self.max_unbounded - 1, maximum_values)

        discrete_observation = np.floor(np.ceil(observation*(self.n_bins/2))/maximum_values) + self.n_bins/2
        discrete_observation = np.where(discrete_observation >= self.n_bins, self.n_bins - 1, discrete_observation)
        discrete_observation = np.where(discrete_observation < 0, 0, discrete_observation)

        return tuple(discrete_observation.astype(int))

    def plot_learning(self):
        import matplotlib.pyplot as plt

        plt.plot(np.arange(0, len(self.episode_rewards)), self.episode_rewards)
        plt.show()
        plt.plot(np.arange(0, len(self.q_sum)), self.q_sum)
        plt.show()

    def print_q_table_info(self):
        print("Values changed? ", (self.q_table != 0).sum(), " out of ", self.q_table.size)

if __name__ == "__main__":
    #simulation params
    NR_EPISODES = 100000
    MAX_T = 200

    #discretisation params
    N_BINS = 12
    MAX_UNBOUNDED = 10

    #Learning params
    DISCOUNT_FACTOR = 1
    MIN_EPSILON = 0.05  #random choice with probability epsilon
    MIN_LEARNING_RATE = 0.02
    LEARNING_DECAY = 0.01

    TEST_SAMPLE = 50
    TEST_SAMPLE_MIN_AVG = 180

    cart_pole = CartPole(MAX_T, N_BINS, MAX_UNBOUNDED, DISCOUNT_FACTOR, MIN_EPSILON,
        MIN_LEARNING_RATE, LEARNING_DECAY, TEST_SAMPLE, TEST_SAMPLE_MIN_AVG)
    cart_pole.run_n_episodes(NR_EPISODES)
    cart_pole.plot_learning()
    cart_pole.print_q_table_info()
