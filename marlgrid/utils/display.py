import matplotlib.pyplot as plt
#%matplotlib inline
from IPython import display
import inspect
import os
import moviepy.video.io.ImageSequenceClip
from pettingzoo.utils.conversions import aec_to_parallel, parallel_to_aec
import datetime
from stable_baselines3.common.results_plotter import load_results, ts2xy
import numpy as np

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_train(log_folder, title='Learning Curve', window=50):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=window)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Timestep')
    plt.ylabel('Reward')
    plt.title(title + " Smoothed")

    plt.savefig(os.path.join(log_folder, title + str(len(x))), bbox_inches='tight')
    #plt.show()

def make_pic_video(model, env, name, savePics, saveVids, savePath, random_policy=False, video_length=50):
    pass
    #stuff provided in video.py... 
    '''print(env.__dict__.keys())
    env = parallel_to_aec(env.unwrapped).unwrapped
    print(env.__dict__.keys())
    vidname = name + '-' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    for i in range(video_length+1):
        if random_policy:
            actions = { a: env.action_spaces[a].sample() for k, a in enumerate(env.possible_agents) }
        else:
            actions = {x: model.predict(obs[x])[0] for x in env.possible_agents}
        obs, rew, dones, _, = env.step(actions)
        ims += [env.render(),]
        if dones['player_0']:
            break
    
    if saveVids:
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(ims, 15)
        clip.write_videofile(os.path.join(savePath , vidname) + '.mp4')'''

def plot_evals(name, stages, rewards, stds, history, savePath, saveEvery=1):
    fig, axs = plt.subplots(1)
    xaxis = range(0,len(rewards[stages[0]])*saveEvery,saveEvery)

    for i, stage in enumerate(stages):

        plt.plot(xaxis, rewards[stage], label=stage, )
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.title(name)
    plt.xlabel('Timestep')
    plt.ylabel('Reward')
    #plt.show()
    plt.savefig(os.path.join(savePath, name), bbox_inches='tight')

def show_state(env, step=0, info=""):
    plt.figure(3)
    display.display(plt.clf())
    plt.imshow(env.render(mode='human'))
    plt.title("%s | Step: %d %s" % (str(env.__class__.__name__), step, info))
    plt.axis('off')

    display.clear_output(wait=True)
    display.display(plt.gcf())

