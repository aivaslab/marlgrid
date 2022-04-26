import matplotlib.pyplot as plt
%matplotlib inline
from IPython import display

def plot_evals(name, stages, rewards, stds):
    fig, axs = plt.subplots(1)
    xaxis = range(len(rewards[stages[0]]))

    for i, stage in enumerate(stages):

        plt.plot(xaxis, rewards[stage], label=stage, )
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.title(name)
    #plt.show()
    plt.savefig("drive/MyDrive/modelEvals/"+name)

def show_state(env, step=0, info=""):
    plt.figure(3)
    display.display(plt.clf())
    plt.imshow(env.render(mode='human'))
    plt.title("%s | Step: %d %s" % (str(env.__class__.__name__), step, info))
    plt.axis('off')

    display.clear_output(wait=True)
    display.display(plt.gcf())