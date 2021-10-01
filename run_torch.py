import numpy as np
from collections import deque
import torch
import os
import random
import tqdm

from IPython import display

import marlgrid
from marlgrid.agents_base import Agent, AC, AC_Deterministic, AC_Network
from marlgrid.agents_torch import ConvLSTMA3C

from marlgrid.rendering import InteractivePlayerWindow
from marlgrid.agents import GridAgentInterface
from marlgrid.envs import env_from_config
from marlgrid.envs.configs import get_config
from src.discrete_A3C import CNet, train_agent
from matplotlib import pyplot as plt
from PIL import Image
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('env')
    parser.add_argument('mode')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    env_name = args.env
    mode = args.mode

    env_config, agent_config = get_config(env_name)
    assert isinstance(agent_config, dict)

    env_config['agents'] = [agent_config]
    env = env_from_config(env_config)

    device = 'cuda:0' #torch.device("cuda")
    channels = 3
    view_size = agent_config['view_size']
    view_tile_size = agent_config['view_tile_size']
    memory_length = 1

    num_workers, max_workers = 0, 12
    num_checkpoints = 1
    model = CNet
    num_actions = 3
    #margs = [channels*memory_length*(view_size*view_tile_size)**2, num_actions]
    margs = [4*4*8, num_actions]
    agent = Agent(AC_Network(model, margs, 1, device=device)) #1 is unused viewsize var
    total_eps = 100
    data_path = 'data'
    agent_name = 'agent' + env_name

    # Train RL Agent
    if mode == 'rl':
        # not os.path.isdir(path) or not os.path.isfile(os.path.join(path, 'a_weights_f.pth')): 
        path = os.path.join(data_path, agent_name)
        if not os.path.isdir(path):
            os.mkdir(path)
        print('training agent', agent_name)
        train_hist = train_agent(agent_name, env, 
                                agent.controller.model, total_eps, 0.9, 
                                model, margs, device, memory_length, 
                                max_workers, num_checkpoints=num_checkpoints, path=path)
        print('avg reward:', train_hist[-1])
        plt.scatter(range(len(train_hist)), train_hist, alpha=0.6)
        plt.savefig( os.path.join(path, 'a_train.png'))

        print('saving weights')
        torch.save(agent.controller.model.state_dict(), os.path.join(path, 'a_weights_f.pth'))

    # Create Supervised Observer Training Data
    elif mode=='data':
        #not os.path.isdir(path) or not os.path.isfile(os.path.join(path,'a_val.pkl')):
        path = os.path.join(data_path, pretrain)
        if not os.path.isdir(path):
            os.mkdir(path)
        print('creating train samples in', path)
        env.full_test = True

        agents[0].controller.model.eval()
        with torch.no_grad():
            train = create_samples(env, agents, numtrainsamples, eps_per_run, None, unarbitrary_prefs=unarbitrary_prefs)
            train_data = format_data_torch(train, 
                        batch_size=batch_size, 
                        shuffle=True, 
                        num_workers=num_workers)
            print('creating val samples')
            env.full_test = False
            val = create_samples(env, agents, numtestsamples, eps_per_run, None, unarbitrary_prefs=unarbitrary_prefs)
            val_data = format_data_torch(val, 
                        batch_size=batch_size, 
                        num_workers=num_workers)

            val_percepts = []
            val_percepts_data = []
            if args.env == 'ge':
                for n in range(num_goals+1):
                    print('creating val_percept', n, 'samples')
                    val_percepts += [create_samples(env, agents, numtestsamples, eps_per_run, 
                                            'percepts', 
                                            unarbitrary_prefs=unarbitrary_prefs, 
                                            num_goals=n, 
                                            rendering=False),]
                    val_percepts_data += [format_data_torch(val_percepts[n], 
                                            batch_size=batch_size, 
                                            num_workers=num_workers),]

        save_data(train, os.path.join(path, 'a_train'))
        save_data(val, os.path.join(path, 'a_val'))

    # Train Supervised Observer
    elif mode=='tomnet':
        print('loading data in', path)
        train = load_data(os.path.join(path, 'a_train'))
        val = load_data(os.path.join(path, 'a_val'))
        train_data = format_data_torch(train, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers)
        val_data = format_data_torch(val, 
            batch_size=batch_size, 
            num_workers=num_workers)
        val_percepts = []
        val_percepts_data = []

        print('training tomnet model at', path)
        num_test_sets = len(val_percepts)
        testsets = [val_data] + [val_percepts_data[n] for n in range(num_test_sets)]
        class Args():
            pass
            
        aargs = Args()
        aargs.log_interval = 2047
        aargs.dry_run = 0
        aargs.gamma = 1
        aargs.epochs = tomnet_epochs

        history = itrain(tomnet, aargs, device, train_data, testsets)
        test_names = ['train','val','p0','p1','p2','p3']

        for item in history:
            plt.plot(item)
        plt.legend(test_names, loc='lower right')
        plt.ylabel('mse')
        plt.xlabel('epoch')
        os.mkdir(path)
        plt.savefig( os.path.join(path, 't_train.png'))
        torch.save(tomnet.state_dict(), os.path.join(path, 't_weights.pth'))


