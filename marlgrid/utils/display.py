import matplotlib.pyplot as plt
#%matplotlib inline
from IPython import display
import inspect
import os
import moviepy.video.io.ImageSequenceClip
import datetime

def make_pic_video(model, env, name, savePics, saveVids, savePath, random_policy=False, video_length=50):
    env = parallel_to_aec(env.unwrapped).unwrapped
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
        clip.write_videofile(os.path.join(savePath , vidname) + '.mp4')

def plot_evals(name, stages, rewards, stds, history, savePath):
    fig, axs = plt.subplots(1)
    xaxis = range(len(rewards[stages[0]]))

    for i, stage in enumerate(stages):

        plt.plot(xaxis, rewards[stage], label=stage, )
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.title(name)
    #plt.show()
    plt.savefig(savePath+name, bbox_inches='tight')

def show_state(env, step=0, info=""):
    plt.figure(3)
    display.display(plt.clf())
    plt.imshow(env.render(mode='human'))
    plt.title("%s | Step: %d %s" % (str(env.__class__.__name__), step, info))
    plt.axis('off')

    display.clear_output(wait=True)
    display.display(plt.gcf())