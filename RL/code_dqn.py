from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from gym import wrappers
import os
import random
import gym
import numpy as np
import pandas as pd

batch_size = 32 # dqn replay batch size
total_episode_count = 25000
render_env_count = 1000

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

model_dir = 'model_output/gym/'
vid_dir = 'vid_output/gym/'
makedir(model_dir)
makedir(vid_dir)

class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.replay_memory = deque(maxlen=2000)

        self.gamma = 0.95 # discount rate
        self.alpha = 0.001
        self.model = self._init_model()

        self.high_epsilon = 1 # for decaying epsilon
        self.low_epsilon = 0.1
        self.epsilon_start_decay = 1
        self.epsilon_end_decay = total_episode_count//2
        self.epsilon = 1
        self.epsilon_decay_value = (self.high_epsilon-self.low_epsilon)/(self.epsilon_end_decay - self.epsilon_start_decay)

        self.df = pd.DataFrame(columns=['episode', 'avg_reward', 'max_reward', 'min_reward','epsilon', 'alive_time'])


    def _init_model(self):
        # NN approximates Q-value
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu')) # states as input
        model.add(Dense(24, activation='relu')) # hidden layer
        model.add(Dense(self.action_size, activation='linear')) # actions as output neurons
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        return model


    def decaying_epsilon(self, episode_count):
        if (self.epsilon_start_decay < episode_count < self.epsilon_end_decay) and (self.low_epsilon <= self.epsilon <= self.high_epsilon):
            self.epsilon -= self.epsilon_decay_value

    def load_model(self, name):
        self.model.load_weights(name)


    def save_model(self, name):
        self.model.save_weights(name)


    def remember(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))


    def get_action(self, state, episode_count):
        self.decaying_epsilon(episode_count)
        if np.random.rand() > self.epsilon:
            action = np.argmax(self.model.predict(state)[0])
        else:
            action = random.randrange(self.action_size)
        return action


    def replay(self, batch_size): # fit NN with updated q values from replay_memory
        replay_sample = random.sample(self.replay_memory, batch_size)

        for state, action, reward, next_state, done in replay_sample:

            if done: # set target as reward received
                target = reward
            elif not done: # predict future discounted reward
                next_state_max_q_value = np.amax(self.model.predict(next_state)[0])
                target = (reward + self.gamma * next_state_max_q_value)

            predicted_target = self.model.predict(state)
            predicted_target[0][action] = target

            self.model.fit(state, predicted_target, epochs=1, verbose=0)


    def record_keeping(self, episode, rewards, timeframe):
        record_dict =  {'episode':episode,'avg_reward':(sum(rewards)/len(rewards)),'max_reward':max(rewards),'min_reward':min(rewards),'total_reward':sum(rewards),'epsilon':self.epsilon,'alive_time':timeframe}
        self.df = self.df.append(record_dict, ignore_index=True)


def main():

    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print(f'state_size: {state_size}, action_size: {action_size}')

    agent = DQN(state_size, action_size)
    env = wrappers.Monitor(env, directory=vid_dir, force=True)

    for episode in range(total_episode_count):

        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        render = episode%render_env_count==0
        episodic_reward = []

        for timeframe in range(1000000): # single frame of game, assuming 1000000 frames to be the limit
            if render:
                env.render()

            action = agent.get_action(state, episode)
            next_state, reward, done, _ = env.step(action)

            reward = reward if not done else -10
            episodic_reward.append(reward)

            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)

            state = next_state

            if done:
                print(f"episode: {episode}/{total_episode_count}, alive_time: {timeframe}, e: {agent.epsilon:.3f}")
                break

        agent.record_keeping(episode, episodic_reward, timeframe)

        if len(agent.replay_memory) > batch_size:
            agent.replay(batch_size)

        if episode % 100 == 0:
            agent.save_model(model_dir + "weights_" + '{:04d}'.format(episode) + ".hdf5")
            agent.df.to_excel('dqn_record_keeping.xlsx', index=None)


    agent.save_model('final_weights.hdf5')
    agent.df.to_excel('dqn_record_keeping.xlsx', index=None)
    env.close()

if __name__ == '__main__':
    main()
