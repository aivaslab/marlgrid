import numpy as np
import marlgrid

from marlgrid.rendering import InteractivePlayerWindow
from marlgrid.agents import GridAgentInterface
from marlgrid.envs import env_from_config
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import random


class dqlPlayer:
    def __init__(self, env):
        self.player_window = InteractivePlayerWindow(
            caption="interactive marlgrid"
        )
        self.episode_count = 0
        self.step_count = 0
        self.env = env

        self.memory  = deque(maxlen=2000)
        
        self.gamma = 0.95
        self.epsilon = 0.9
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.tau = .05
        self.observation_space = (-1,5,3)
        self.action_space = np.array(range(3))
        self.model = self.create_model()
        # "hack" implemented by DeepMind to improve convergence
        self.target_model = self.create_model()
        self.batch_size = 64

    def create_model(self):
        model   = Sequential()
        state_shape  = self.observation_space
        model.add(Dense(self.action_space.shape[0], input_shape=(5,5,3), activation="linear"))
        #model.add(Dense(24, activation="relu"))
        #model.add(Dense(24, activation="relu"))
        #model.add(Dense(self.action_space.shape[0]))
        #print(self.action_space.shape[0])
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate))
        return model

    def action_step(self, obs):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space,1)[0]
        return np.argmax(self.model.predict(obs.astype(np.uint8))[0][0])

    def save_step(self, obs, act, rew, new_obs, done):
        self.step_count += 1

        if rew == 0:
            rew = -1
        self.cumulative_reward += rew
        print(f"   step {self.step_count:<4d}: reward {rew} (episode total {self.cumulative_reward}) epsilon {self.epsilon}")
        

        self.memory.append([obs, act, rew, new_obs, done])

        if done and self.step_count >= self.batch_size: #(self.episode_count + self.step_count) % self.batch_size == 0:
            #print('training')
            self.train2(rand=True)

    def start_episode(self):
        self.cumulative_reward = 0
        self.step_count = 0
    
    def end_episode(self):
        print(
            f"Finished episode {self.episode_count} after {self.step_count} steps."
            f"  Episode return was {self.cumulative_reward}."
        )
        self.episode_count += 1

    def train(self, rand=False):
        if random:
            minibatch = random.sample(self.memory, self.batch_size)
        else:
            minibatch = self.memory[:self.batch_size]
        #print(minibatch)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            #print('x', np.array(next_state).shape, np.array(state).shape)
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def train2(self, rand=False):
        minibatch = random.sample(self.memory, self.batch_size)
        state_vec = []
        target_vec = []

        for state, action, reward, next_state, done in minibatch:
            #print('state',state)
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            state_vec += [state,]
            target_vec += [target_f,]
        #print(len(state_vec))
        state_vec = np.asarray(state_vec).astype(np.float32)
        target_vec = np.asarray(target_vec).astype(np.float32)
        print(state_vec.shape)
        self.model.fit(state_vec, target_vec, epochs=1, verbose=1)
        #self.model.fit(np.squeeze(np.array(state_vec)), np.squeeze(np.array(target_vec)), epochs=1, verbose=0)



'''env_config =  {
    "env_class": "ClutteredGoalCycleEnv",
    "grid_size": 13,
    "max_steps": 250,
    "clutter_density": 0.15,
    "respawn": True,
    "ghost_mode": True,
    "reward_decay": False,
    "n_bonus_tiles": 3,
    "initial_reward": True,
    "penalty": -1.5
}'''

env_config =  {
    "env_class": "EmptyMultiGrid",
    "grid_size": 6,
    "max_steps": 100,
    "respawn": True,
    "ghost_mode": True,
    "reward_decay": False,
}

player_interface_config = {
    "view_size": 5,
    "view_offset": 1,
    "view_tile_size": 1,
    "observation_style": "rich",
    "see_through_walls": False,
    "color": "prestige"
}

env_config['agents'] = [player_interface_config]

env = env_from_config(env_config)

human = dqlPlayer(env)
total_eps = 100
for batch in range(total_eps):
    obs_list = env.reset()

    human.start_episode()
    done = False
    while not done:

        env.render() # OPTIONAL: render the whole scene + birds eye view
        #print('obs:',obs_list[0])
        #print('d')
        player_action = human.action_step(obs_list[0]['pov'])
        # The environment expects a list of actions, so put the player action into a list
        agent_actions = [player_action]

        next_obs_list, rew_list, done, _ = env.step(agent_actions)
        
        human.save_step(
            obs_list[0]['pov'], player_action, rew_list[0], next_obs_list[0]['pov'], done
        )

        obs_list = next_obs_list

    human.end_episode()