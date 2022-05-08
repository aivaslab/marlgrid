import matplotlib.pyplot as plt
#%matplotlib inline
from IPython import display
import inspect
import os
import moviepy.video.io.ImageSequenceClip
from pettingzoo.utils.conversions import aec_to_parallel, parallel_to_aec
import datetime
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import get_monitor_files
import numpy as np
import pandas

def load_results_tempfix(path: str) -> pandas.DataFrame:
    # something causes broken csvs, here we ignore extra data
    monitor_files = get_monitor_files(path)
    if len(monitor_files) == 0:
        raise LoadMonitorResultsError(f"No monitor files of the form *{Monitor.EXT} found in {path}")
    data_frames, headers = [], []
    for file_name in monitor_files:
        with open(file_name) as file_handler:
            first_line = file_handler.readline()
            assert first_line[0] == "#"
            header = json.loads(first_line[1:])
            #cols = pandas.read_csv(file_handler, nrows=1).columns
            data_frame = pandas.read_csv(file_handler, index_col=None, on_bad_lines='skip')#, usecols=cols)
            headers.append(header)
            data_frame["t"] += header["t_start"]
        data_frames.append(data_frame)
    data_frame = pandas.concat(data_frames)
    data_frame.sort_values("t", inplace=True)
    data_frame.reset_index(inplace=True)
    data_frame["t"] -= min(header["t_start"] for header in headers)
    return data_frame

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
    x, y = ts2xy(load_results_tempfix(log_folder), 'timesteps')
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

def plot_evals(savePath, name, names, eval_cbs):
    fig, axs = plt.subplots(1)
    for env_name, cb in zip(names, eval_cbs):
        plt.plot(cb.evaluations_timesteps, [np.mean(x) for x in cb.evaluations_results], label=env_name, )
        plt.scatter(cb.evaluations_timesteps, cb.evaluations_results, label=env_name, )
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.title(name)
    plt.xlabel('Timestep')
    plt.ylabel('Reward')
    print(savePath, name)
    plt.savefig(os.path.join(savePath, name+'_evals'), bbox_inches='tight')

def plot_evals_legacy(name, stages, rewards, stds, history, savePath, saveEvery=1):
    fig, axs = plt.subplots(1)
    xaxis = range(0,len(rewards[stages[0]])*saveEvery,saveEvery)

    for i, stage in enumerate(stages):

        plt.plot(xaxis, rewards[stage], label=stage, )
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.title(name)
    plt.xlabel('Timestep')
    plt.ylabel('Reward')
    plt.savefig(os.path.join(savePath, name), bbox_inches='tight')

def show_state(env, step=0, info=""):
    plt.figure(3)
    display.display(plt.clf())
    plt.imshow(env.render(mode='human'))
    plt.title("%s | Step: %d %s" % (str(env.__class__.__name__), step, info))
    plt.axis('off')

    display.clear_output(wait=True)
    display.display(plt.gcf())

