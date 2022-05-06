import os
from .conversion import make_env
from .display import make_pic_video, plot_evals, plot_train
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, EveryNTimesteps, BaseCallback
import tqdm

class PlottingCallback(BaseCallback):
    """
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, verbose=0, savePath='', name='', envs=[], names=[]):
        super(PlottingCallback, self).__init__(verbose)
        self.savePath = savePath
        self.name = name
        self.envs = []
        self.names = []

    def _on_step(self) -> bool:
        #plot things
        #plot_train(self.savePath, self.name)
            
        for env, name in zip(self.envs, self.names):
            make_pic_video(self.model, env, name, False, True, self.savePath)
        return True

def train_model(name, train_env, eval_envs, eval_params,
                player_config,
                framework, policy, learning_rate=1e-5, 
                evals=25, total_timesteps=1e6, eval_eps=25,
                batch_size=32, memory=1, size=64, reduce_color=False,
                threads=1, saveModel=False, saveVids=True, savePics=True, 
                saveEval=True, saveTrain=True,
                savePath="drive/MyDrive/model/", reward_decay=True,
                extractor_features=32):

    if saveModel or saveVids or savePics:
        savePath = os.path.join(savePath, name)
        if not os.path.exists(savePath):
            os.mkdir(savePath)

    recordEvery = int(total_timesteps/evals) if evals > 0 else 1000
    
    train_env = make_env(train_env[0], player_config, train_env[1], memory=memory, threads=threads,
                         reduce_color=reduce_color, size=size, reward_decay=reward_decay,
                         path=savePath)

    policy_kwargs = dict(
        #change model hyperparams
        features_extractor_kwargs=dict(features_dim=extractor_features),
        )
    #model = framework(policy, train_env, learning_rate=learning_rate, n_steps=batch_size, tensorboard_log=logdir, use_rms_prop=True)
    model = framework(policy, train_env, learning_rate=learning_rate, 
                      n_steps=batch_size, tensorboard_log="logs", policy_kwargs=policy_kwargs)
    eval_envs = [make_env(x, player_config, y, memory=memory, threads=threads, 
                          reduce_color=reduce_color, size=size, saveVids=saveVids, path=savePath, 
                          recordEvery=recordEvery, reward_decay=reward_decay) for x,y in 
                          zip(eval_envs, eval_params)]
    name = str(name+model.policy_class.__name__)

    plot_cb = PlottingCallback(savePath, name, eval_envs, eval_params)
    eval_cbs = [EvalCallback(eval_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=1,
                             n_eval_episodes=eval_eps,
                             deterministic=True, render=False) for eval_env in eval_envs]
    eval_cbs.append(plot_cb)

    cb = [EveryNTimesteps(n_steps=recordEvery, callback=CallbackList(eval_cbs,verbose=0))]

    model.learn(total_timesteps=total_timesteps, 
                tb_log_name=name, reset_num_timesteps=True, callback=cb)
    plot_train(savePath, name+'train')

    if saveModel:
        model.save(os.path.join(savePath, name))
    return train_env
