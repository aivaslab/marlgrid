import os
from .conversion import make_env
from .display import make_pic_video, plot_evals, plot_train
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, EveryNTimesteps, BaseCallback
from tqdm.notebook import tqdm
import logging

class TqdmCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.progress_bar = None
    
    def _on_training_start(self):
        self.progress_bar = tqdm(total=self.locals['total_timesteps'])
    
    def _on_step(self):
        self.progress_bar.update(1)
        return True

    def _on_training_end(self):
        self.progress_bar.close()
        self.progress_bar = None

class PlottingCallback(BaseCallback):
    """
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, verbose=0, savePath='', name='', envs=[], names=[], eval_cbs=[]):
        super(PlottingCallback, self).__init__(verbose)
        self.savePath = savePath
        self.name = name
        self.envs = envs
        self.names = names
        self.eval_cbs = eval_cbs

    def _on_step(self) -> bool:
        plot_evals(self.savePath, self.name, self.names, self.eval_cbs)
        for env, name in zip(self.envs, self.names):
            make_pic_video(self.model, env, name, False, True, self.savePath)
        return True

class PlottingCallbackStartStop(BaseCallback):
    """
    #bandaid fix to plotting not happening at training start and end
    """
    def __init__(self, verbose=0, savePath='', name='', envs=[], names=[], eval_cbs=[]):
        super(PlottingCallbackStartStop, self).__init__(verbose)
        self.savePath = savePath
        self.name = name
        self.envs = envs
        self.names = names
        self.eval_cbs = eval_cbs

    def _on_training_start(self) -> bool:
        super(PlottingCallbackStartStop, self)._on_training_start()
        plot_evals(self.savePath, self.name, self.names, self.eval_cbs)
        for env, name in zip(self.envs, self.names):
            make_pic_video(self.model, env, name, False, True, self.savePath)
        return True

    def _on_step(self) -> bool:
        super(PlottingCallbackStartStop, self)._on_step()

    def _on_training_end(self) -> bool:
        super(PlottingCallbackStartStop, self)._on_training_end()
        plot_evals(self.savePath, self.name, self.names, self.eval_cbs)
            
        for env, name in zip(self.envs, self.names):
            make_pic_video(self.model, env, name, False, True, self.savePath)
        return True

def train_model(name, train_env, eval_envs, eval_params,
                player_config,
                framework, policy, learning_rate=1e-5, 
                evals=25, total_timesteps=1e6, eval_eps=25,
                batch_size=32, memory=1, size=64, reduce_color=False,
                threads=1, saveModel=True, saveVids=True, savePics=True, 
                saveEval=True, saveTrain=True,
                savePath="drive/MyDrive/model/", reward_decay=True,
                extractor_features=32):

    if not os.path.exists(savePath):
        os.mkdir(savePath)
    savePath = os.path.join(savePath, name)
    if not os.path.exists(savePath):
        os.mkdir(savePath)

    #log all hyperparameters to a file
    with open(os.path.join(savePath, 'logs.txt'), 'w') as logfile:
        logfile.write(str(locals()))

    recordEvery = int(total_timesteps/evals) if evals > 0 else 1000
    
    train_env = make_env(train_env[0], player_config, train_env[1], memory=memory, threads=threads,
                         reduce_color=reduce_color, size=size, reward_decay=reward_decay,
                         path=savePath)

    policy_kwargs = dict(
        #change model hyperparams
        features_extractor_kwargs=dict(features_dim=extractor_features ),
        ) if policy[0]=='C' else {}
    #model = framework(policy, train_env, learning_rate=learning_rate, n_steps=batch_size, tensorboard_log=logdir, use_rms_prop=True)
    model = framework(policy, train_env, learning_rate=learning_rate, 
                      n_steps=batch_size, tensorboard_log="logs", policy_kwargs=policy_kwargs)
    eval_envs = [make_env(x, player_config, y, memory=memory, threads=threads, 
                          reduce_color=reduce_color, size=size, saveVids=saveVids, path=savePath, 
                          recordEvery=recordEvery, reward_decay=reward_decay) for x,y in 
                          zip(eval_envs, eval_params)]
    name = str(name+model.policy_class.__name__)

    eval_cbs = [EvalCallback(eval_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=1,
                             n_eval_episodes=eval_eps,
                             deterministic=True, render=False, verbose=0) for eval_env in eval_envs]
    plot_cb = PlottingCallback( verbose=0, savePath=savePath, 
        name=name, envs=eval_envs, names=eval_params, eval_cbs=eval_cbs)
    plot_cb_2 = PlottingCallbackStartStop(verbose=0, savePath=savePath, 
        name=name, envs=eval_envs, names=eval_params, eval_cbs=eval_cbs)
    eval_cbs.append(plot_cb)

    cb = [EveryNTimesteps(n_steps=recordEvery, callback=CallbackList(eval_cbs)), TqdmCallback(), plot_cb_2]

    model.learn(total_timesteps=total_timesteps, 
                tb_log_name=name, reset_num_timesteps=True, callback=cb)

    plot_train(savePath, name+'train')

    if saveModel:
        model.save(os.path.join(savePath, name))
    return train_env
