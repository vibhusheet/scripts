import os
import gym
from gym import wrappers
import numpy as np
import pandas as pd

total_episode_count = 25000
render_env_count = 100

vid_dir = 'vid_output/gym/'

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

makedir(vid_dir)



class q_learning():

    def __init__(self, env):
        self.action_space_size = env.action_space.n
        self.env_observation_space_low = env.observation_space.low
        self.env_observation_space_high = env.observation_space.high

        self.obs_discrete_size = [20] * env.observation_space.shape[0]
        self.discrete_obs_bucket_size = (env.observation_space.high - env.observation_space.low)/self.obs_discrete_size

        # initializing random q_values between (-2, 0)
        self.q_table = np.random.uniform(low=-2, high=0, size=(self.obs_discrete_size+[self.action_space_size]))

        self.alpha = 0.1 # learning rate
        self.gamma = 0.95 # discount factor

        self.high_epsilon = 1 # for decaying epsilon
        self.low_epsilon = 0.1
        self.epsilon_start_decay = 1
        self.epsilon_end_decay = total_episode_count//2
        self.epsilon = 1
        self.epsilon_decay_value = (self.high_epsilon-self.low_epsilon)/(self.epsilon_end_decay - self.epsilon_start_decay)

        self.df = pd.DataFrame(columns=['episode', 'avg_reward', 'max_reward', 'min_reward','epsilon'])


    def discrete_state(self, state):
        #bucketing continuous states into discrete values
        _discrete_state = (state - self.env_observation_space_low)/self.discrete_obs_bucket_size
        return tuple(_discrete_state.astype(np.int))


    def decaying_epsilon(self, episode_count):
        if (self.epsilon_start_decay < episode_count < self.epsilon_end_decay) and (self.low_epsilon <= self.epsilon <= self.high_epsilon):
            self.epsilon -= self.epsilon_decay_value


    def get_action(self, state, episode_count):
        self.decaying_epsilon(episode_count)
        if np.random.random() > self.epsilon:
            action = np.argmax(self.q_table[state])
        else:
            action = np.random.randint(0, self.action_space_size)
        return action


    def update_q_table(self, state, new_state, action, reward):
        q_value = self.q_table[state + (action,)]
        next_max_q_value = np.max(self.q_table[new_state])

        update_q_value = (1-self.alpha) * q_value + self.alpha * (reward + self.gamma * next_max_q_value)
        self.q_table[state + (action,)] = update_q_value


    def record_keeping(self, episode, rewards):
        record_dict =  {'episode':episode,'avg_reward':(sum(rewards)/len(rewards)),'max_reward':max(rewards),'min_reward':min(rewards),'total_reward':sum(rewards),'epsilon':self.epsilon}
        self.df = self.df.append(record_dict, ignore_index=True)


def main():
    env = gym.make('MountainCar-v0')

    env = wrappers.Monitor(env, directory=vid_dir, force=True)

    q_learning_agent = q_learning(env)

    for episode in range(total_episode_count):
        discrete_state = q_learning_agent.discrete_state(env.reset())
        done = False
        render = episode%render_env_count==0
        episodic_reward = []

        if episode%10==0:
            print(f'episode reached: {episode}')

        while not done:
            action = q_learning_agent.get_action(discrete_state, episode)
            next_env_state, reward, done, info = env.step(action)
            new_discrete_state = q_learning_agent.discrete_state(next_env_state)
            episodic_reward.append(reward)

            if render:
                env.render()

            if not done:
                q_learning_agent.update_q_table(discrete_state, new_discrete_state, action, reward)

            elif (next_env_state[0] >= env.goal_position):
                print(f'achieved goal_state at {episode}')
                q_learning_agent.q_table[discrete_state + (action,)] = 0

            discrete_state = new_discrete_state

        q_learning_agent.record_keeping(episode, episodic_reward)

    q_learning_agent.df.to_excel('q_learning_record_keeping.xlsx', index=None)
    env.close()


if __name__=='__main__':
    main()
