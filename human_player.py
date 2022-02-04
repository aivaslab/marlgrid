import numpy as np
import marlgrid

from marlgrid.rendering import InteractivePlayerWindow
from marlgrid.agents import GridAgentInterface
#from marlgrid.envs import env_from_config
from marlgrid.pz_envs import *

class HumanPlayer:
    def __init__(self):
        self.player_window = InteractivePlayerWindow(
            caption="interactive marlgrid"
        )
        self.episode_count = 0

    def action_step(self, obs):
        return self.player_window.get_action(obs.astype(np.uint8))

    def save_step(self, obs, act, rew, done):
        print(f"   step {self.step_count:<3d}: reward {rew} (episode total {self.cumulative_reward})")
        self.cumulative_reward += rew
        self.step_count += 1

    def start_episode(self):
        self.cumulative_reward = 0
        self.step_count = 0
    
    def end_episode(self):
        print(
            f"Finished episode {self.episode_count} after {self.step_count} steps."
            f"  Episode return was {self.cumulative_reward}."
        )
        self.episode_count += 1


env_config =  {
    "env_class": "para_TutorialEnv",
    "max_steps": 250,
    "respawn": True,
    "ghost_mode": True,
    "reward_decay": False,
    "width": 9,
    "height": 9
}

player_interface_config = {
    "view_size": 7,
    "view_offset": 1,
    "view_tile_size": 32,
    "observation_style": "rich",
    "see_through_walls": False,
    "color": "prestige"
}

env_config['agents'] = [GridAgentInterface(**player_interface_config)]

env = env_from_config(env_config)



human = HumanPlayer()

human.start_episode()
done = False
for i in range(5):
    obs = env.reset()
    while True:

        #env.unwrapped.render() # OPTIONAL: render the whole scene + birds eye view

        player_action = human.action_step(obs['player_0']['pov'])

        agent_actions = {'player_0': player_action}

        next_obs, rew, done, _ = env.step(agent_actions)
        reward = rew['player_0']
        
        
        human.save_step(
            obs['player_0'], player_action, reward, done
        )

        obs = next_obs

        if done['player_0'] == True:
            break

human.end_episode()
