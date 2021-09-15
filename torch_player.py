import numpy as np
import marlgrid
from marlgrid.agents_base import Agent, AC, AC_Deterministic, AC_Network
from marlgrid.agents_torch import ConvLSTMA3C

from marlgrid.rendering import InteractivePlayerWindow
from marlgrid.agents import GridAgentInterface
from marlgrid.envs import env_from_config
from collections import deque
import torch
import os
import random
import tqdm
from src import discrete_A3C
from matplotlib import pyplot as plt
from PIL import Image


if __name__ == '__main__':


    env_config0 =  {
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
    }

    env_config1 =  {
        "env_class": "EmptyMultiGrid",
        "grid_size": 6,
        "max_steps": 500,
        "respawn": True,
        "ghost_mode": True,
        "reward_decay": False,
    }

    env_config2 =  {
        "env_class": "DoorKeyEnv",
        "grid_size": 6,
        "max_steps": 250,
        "respawn": True,
        "ghost_mode": True,
        "reward_decay": False,
    }

    env_config3 =  {
        "env_class": "ContentFBEnv",
        "grid_size": 15,
        "max_steps": 250,
        "respawn": True,
        "ghost_mode": True,
        "reward_decay": True,
    }

    player_interface_config = {
        "view_size": 6,
        "view_offset": 1,
        "view_tile_size": 5,
        "observation_style": "image",
        "see_through_walls": False,
        "color": "prestige"
    }

    env_config = env_config1 # select from above

    env_config['agents'] = [player_interface_config]

    env = env_from_config(env_config)

    #human = dqlPlayer(env)
    device = 'cuda:0' #torch.device("cuda")
    channels = 3
    view_size = 6*5
    memory_length = 1

    max_workers = 14
    num_workers = 0 # 0 for automatic
    num_checkpoints = 1
    model = discrete_A3C.Net
    margs = [channels*view_size*view_size*memory_length, 5]
    human = Agent(AC_Network(model, margs, view_size=(6, 6), device=device))
    total_eps = 10000
    data_path = 'data'
    agent_name = 'agent'

    if True: #agents[0].controller.trainable:
        path = os.path.join(data_path, agent_name)
        if not os.path.isdir(path) or not os.path.isfile(os.path.join(path, 'a_weights_f.pth')): 
            if not os.path.isdir(path):
                os.mkdir(path)
            print('training agent', agent_name)
            train_hist = discrete_A3C.train_agent(agent_name, env, 
                                    human.controller.model, total_eps, 0.9, 
                                    model, margs, device, memory_length, 
                                    max_workers, num_checkpoints=num_checkpoints, path=path)
            print('avg reward:', train_hist[-1])
            plt.scatter(range(len(train_hist)), train_hist, alpha=0.6)
            plt.savefig( os.path.join(path, 'a_train.png'))

            print('saving weights')
            torch.save(human.controller.model.state_dict(), os.path.join(path, 'a_weights_f.pth'))




    #print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    '''
    for batch in range(total_eps):
        obs_list = env.reset()

        human.start_episode()
        done = False
        while not done:

            #env.render() # OPTIONAL: render the whole scene + birds eye view
            #print('d')
            if player_interface_config['observation_style'] == 'rich':
                ob = obs_list[0]['pov']
            else:
                ob = obs_list[0][np.newaxis, :]
            player_action = human.action_step(ob)

            agent_actions = [player_action]

            next_obs_list, rew_list, done, _ = env.step(agent_actions)
            
            if player_interface_config['observation_style'] == 'rich':
                human.save_step(
                    ob, player_action, rew_list[0], next_obs_list[0]['pov'], done
                )
            else:
                human.save_step(
                    ob, player_action, rew_list[0], next_obs_list[0][np.newaxis, :], done
                )

            obs_list = next_obs_list

        human.end_episode()'''