import gym
import json
import datetime as dt
from stable_baselines3 import PPO, A2C
from marlgrid.envs import env_from_config
from marlgrid.envs.configs import get_config
#from stable_baselines3.common.policies import MultiInputActorCriticPolicy

from stable_baselines3.common.envs import SimpleMultiObsEnv
from stable_baselines3.common.vec_env import DummyVecEnv


# Stable Baselines provides SimpleMultiObsEnv as an example environment with Dict observations
env = SimpleMultiObsEnv(random_start=False)

env_name='d'
env_config, agent_config = get_config(env_name)
assert isinstance(agent_config, dict)
env_config['agents'] = [agent_config]
# The algorithms require a vectorized environment to run
env = env_from_config(env_config)
#env = DummyVecEnv([lambda: env_from_config(env_config)])

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=1e3)



env_name='d'
env_config, agent_config = get_config(env_name)
assert isinstance(agent_config, dict)
env_config['agents'] = [agent_config]
# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: env_from_config(env_config)])

model = A2C(MultiInputActorCriticPolicy, env, verbose=1)
model.learn(total_timesteps=20000)
obs = env.reset()
print(obs)
for i in range(2000):
  action, _states = model.predict(obs)
  obs, rewards, done, info = env.step(action)
  env.render()