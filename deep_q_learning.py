import gym
import numpy as np
import math
from collections import deque

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import backend

import random

class CartPole():
    def __init__(self, max_t, discount_factor, min_epsilon, min_learning_rate,
        learning_decay, test_sample, test_sample_min_avg, memory_buffer, random_exploration_steps):
        #game environment
        self.env = gym.make('CartPole-v0')
        #number of frames required to win the game
        self.max_t = max_t
        #discounts future anticipated reward
        self.discount_factor = discount_factor
        #minimum value of epsilon, which is the probability of choosing a random action
        self.min_epsilon = min_epsilon
        #minimum value of the learning rate
        self.min_learning_rate = min_learning_rate
        #decay constants of learning rate wrt time
        self.learning_decay = learning_decay
        #number of episodes before target network is updated
        self.target_update_interval = 20
        #used to end training early when sufficient process is made
        self.test_sample = test_sample
        self.test_sample_min_avg = test_sample_min_avg
        self.training_sample_size = 32

        self.random_exploration_steps = random_exploration_steps
        self.step_counter = 0
        #progress tracking
        self.episode_rewards = []
        self.q_sum = []
        self.criterion = MeanSquaredError()
        self.optimizer = Adam()
        #initialise model
        self.prediction_model = self.init_model()
        self.target_model = self.init_model()

        #initialise memory
        self.memory = deque(maxlen=memory_buffer)

    def init_model(self):
        '''
        Initialise the NN that estimates Q from state
        '''
        model = Sequential()
        model.add(Flatten(input_shape=(4,)))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(self.env.action_space.n))
        model.add(Activation('linear'))

        model.compile(optimizer=self.optimizer, loss=MeanSquaredError())

        return model

    def update_target_model(self):
        '''
        Sets target NN equal to prediction NN
        '''
        self.target_model.set_weights(self.prediction_model.get_weights())

    def update_prediction_model(self):
        '''
        '''
        #if not enough samples yet
        if len(self.memory) < self.training_sample_size: return

        sample = np.array(random.sample(self.memory, self.training_sample_size)).T

        sample = np.stack(sample)
        s_cur_batch, s_new_batch, action_batch, reward_batch, done_batch = sample

        s_cur_batch = np.stack(s_cur_batch)
        s_new_batch = np.stack(s_new_batch)

        done_batch = done_batch.astype(float)

        #use target network to obtain target values for q
        future_reward_batch = self.target_model.predict(s_new_batch)

        target_q_batch = reward_batch + self.discount_factor * tf.reduce_max(
            future_reward_batch, axis = 1
        )

        # If final frame set last q value to -1
        target_q_batch = target_q_batch * (1 - done_batch) - done_batch

        #mask allows us to discard loss for action that was NOT taken
        masks = tf.one_hot(action_batch, self.env.action_space.n)

        with tf.GradientTape() as tape:
            q_hat_batch = self.prediction_model(s_cur_batch)
            #apply mask to get Q-value for action taken
            q_hat_action_batch = tf.reduce_sum(tf.multiply(q_hat_batch, masks), axis = 1)
            # calc loss
            loss = self.criterion(target_q_batch, q_hat_action_batch)

            #Adjust weights
            gradients = tape.gradient(loss, self.prediction_model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.prediction_model.trainable_variables))

    def run_single_episode(self, render = False):
        '''
        Execute a single episode of the game.
        ====================================
        PARAMS:
            render : Boolean
                -determines if simulation is rendered for human viewing
        RETURNS:
            reward: int
                -reward points
            t+1 : int
                -termination time
        '''
        self.step_counter += 1

        current_state = self.env.reset() #initial observation
        for t in range(self.max_t):
            if render:
                self.env.render()
            action = self.choose_action(current_state)
            #get new state
            new_state, reward, done, _ = self.env.step(action)
            #append this step to memory
            self.memory.append((current_state, new_state, action, reward, done))
            self.update_prediction_model()
            current_state = new_state
            if done:
                return reward, t+1

    def run_n_episodes(self, n):
        for episode_nr in range(n):

            if episode_nr % self.target_update_interval == 0:
                self.update_target_model()
                print("episode {}".format(episode_nr))

            #get current learning rate, epsilon
            self.epsilon = self.get_epsilon(episode_nr)
            self.learning_rate = self.get_learning_rate(episode_nr)
            reward, termination_time = self.run_single_episode()
            self.episode_rewards.append(termination_time)
            #print(termination_time)
            #training completed
            if np.average(self.episode_rewards[-self.test_sample:]) > self.test_sample_min_avg:
                break

        #print("FINISHED TRAINING")

        #showcase a small number of rendered simulations
        for showcase_nr in range(5):
            self.run_single_episode(render=True)

        self.env.close()

    def get_epsilon(self, t):
        '''
        Decaying epsilon
        '''
        return max(self.min_epsilon, np.exp(-self.learning_decay*t))

    def get_learning_rate(self, t):
        '''
        Decaying learning rate
        '''
        return max(self.min_learning_rate, np.exp(-self.learning_decay*t))

    def choose_action(self, state):
        '''
        Choose action based on epsilon greedy policy:
        higher predicted q-value or with probability epsilon a
        random action
        ====================================================================
        PARAMS:
            state : np.array
                -the current state of the simulation based on which an action
                 is taken

        RETURNS:
            0 or 1 (left or right input)
        '''
        if np.random.random() <= self.epsilon or self.step_counter < self.random_exploration_steps:
            return self.env.action_space.sample()
        else:
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)
            return tf.argmax(self.prediction_model(state, training=False)[0]).numpy()

    def plot_learning(self):
        import matplotlib.pyplot as plt

        plt.plot(np.arange(0, len(self.episode_rewards)), self.episode_rewards)
        plt.show()
        plt.plot(np.arange(0, len(self.q_sum)), self.q_sum)
        plt.show()


if __name__ == "__main__":
    #simulation params
    NR_EPISODES = 100000
    MAX_T = 500

    #Learning params
    DISCOUNT_FACTOR = 1
    MIN_EPSILON = 0.08  #random choice with probability epsilon
    MIN_LEARNING_RATE = 0.03
    LEARNING_DECAY = 0.01
    RANDOM_EXPLORATION_STEPS = 100

    TEST_SAMPLE = 10
    TEST_SAMPLE_MIN_AVG = 180

    #Memory buffer
    MEMORY_BUFFER = int(4e4) #nr entries in memory

    cart_pole = CartPole(MAX_T, DISCOUNT_FACTOR, MIN_EPSILON, MIN_LEARNING_RATE,
        LEARNING_DECAY, TEST_SAMPLE, TEST_SAMPLE_MIN_AVG, MEMORY_BUFFER, RANDOM_EXPLORATION_STEPS)
    cart_pole.run_n_episodes(NR_EPISODES)
    cart_pole.plot_learning()
