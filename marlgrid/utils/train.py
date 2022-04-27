import os
from .conversion import make_env
from .display import make_pic_video, plot_evals, plot_train
from stable_baselines3.common.evaluation import evaluate_policy
import tqdm

def evaluate_all_levels(model, eval_envs, eval_names, rewards, stds, n_eval_episodes=20, 
                        deterministic=True, saveVids=None, 
                        savePics=None, savePath=None, name=None):
    
    for env, name in zip(eval_envs, eval_names):
        #todo: reset env with set seed to be deterministic if it isnt

        if savePics or saveVids:
            make_pic_video(model, env, name, savePics, saveVids, savePath)

        mean_reward, std_reward = evaluate_policy(model=model, env=env, 
                                    n_eval_episodes=n_eval_episodes, 
                                    deterministic=deterministic)
        if name not in rewards.keys():
            rewards[name] = []
            stds[name] = []
        rewards[name] += [mean_reward,]
        stds[name] +=  [std_reward, ]

def train_model(name, train_env, eval_envs, eval_params,
                player_config,
                framework, policy, learning_rate=1e-5, 
                evals=25, total_timesteps=1e6, eval_eps=25,
                batch_size=32, memory=1, size=64, reduce_color=False,
                threads=1, saveModel=False, saveVids=True, savePics=True, 
                saveEval=True, saveTrain=True,
                savePath="drive/MyDrive/model/", reward_decay=True):

    
    if saveModel or saveVids or savePics:
        savePath = os.path.join(savePath, name)
        if not os.path.exists(savePath):
            os.mkdir(savePath)

    recordEvery = int(total_timesteps/evals)
    
    train_env = make_env(train_env[0], player_config, train_env[1], memory=memory, threads=threads,
                         reduce_color=reduce_color, size=size, reward_decay=reward_decay)

    #model = framework(policy, train_env, learning_rate=learning_rate, n_steps=batch_size, tensorboard_log=logdir, use_rms_prop=True)
    model = framework(policy, train_env, learning_rate=learning_rate, 
                      n_steps=batch_size, tensorboard_log="logs")
    eval_envs = [make_env(x, player_config, y, memory=memory, threads=threads, 
                          reduce_color=reduce_color, size=size, saveVids=saveVids, path=savePath, 
                          recordEvery=recordEvery, reward_decay=reward_decay) for x,y in 
                          zip(eval_envs, eval_params)]
    name = str(name+model.policy_class.__name__)
    rewards, stds = {}, {}
    evaluate_all_levels(model, eval_envs, eval_params, rewards, stds, 
                        n_eval_episodes=eval_eps, deterministic=True,
                        saveVids=saveVids, savePics=savePics, savePath=savePath,
                        name=name)

    histories = []
    for step in tqdm.tqdm(range(evals)):
        history = model.learn(total_timesteps=recordEvery, 
                              tb_log_name=name, reset_num_timesteps=True)
        histories += [history,]
        evaluate_all_levels(model, eval_envs, eval_params, rewards, stds, 
                            n_eval_episodes=eval_eps, deterministic=True, 
                            saveVids=saveVids, savePics=savePics, 
                            savePath=savePath, name=name)
        if saveEval:
            plot_evals(name+"_eval", eval_params, rewards, stds, history, 
                       savePath=savePath)
        if saveTrain:
            plot_train(name+"_train", history, savePath=savePath)

    if saveModel:
        model.save(os.path.join(savePath, name))
    return rewards, stds, histories