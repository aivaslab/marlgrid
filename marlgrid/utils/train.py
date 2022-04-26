import os
from .conversion import make_env
from .display import make_pic_video, plot_evals
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
                savePath="drive/MyDrive/model/"):
    
    if saveModel or saveVids or savePics:
        if not os.path.exists(savePath):
            os.mkdir(savePath)
    
    train_env = make_env(train_env[0], player_config, train_env[1], memory=memory, threads=threads,
                         reduce_color=reduce_color, size=size)

    logdir = str(name) + str(framework)
    #model = framework(policy, train_env, learning_rate=learning_rate, n_steps=batch_size, tensorboard_log=logdir, use_rms_prop=True)
    model = framework(policy, train_env, learning_rate=learning_rate, 
                      n_steps=batch_size, tensorboard_log=logdir)
    eval_envs = [make_env(x, player_config, y, memory=memory, threads=threads, 
                          reduce_color=reduce_color, size=size) for x,y in 
                          zip(eval_envs, eval_params)]
    name = str(name+model.policy_class.__name__)
    rewards, stds = {}, {}
    evaluate_all_levels(model, eval_envs, eval_params, rewards, stds, 
                        n_eval_episodes=eval_eps, deterministic=True,
                        saveVids=saveVids, savePics=savePics, savePath=savePath,
                        name=name)

    histories = []
    for step in tqdm.tqdm(range(evals)):
        history = model.learn(total_timesteps=int(total_timesteps/evals), 
                              tb_log_name=name, reset_num_timesteps=True)
        histories += [history,]
        evaluate_all_levels(model, eval_envs, eval_params, rewards, stds, 
                            n_eval_episodes=eval_eps, deterministic=True, 
                            saveVids=saveVids, savePics=savePics, 
                            savePath=savePath, name=name)
        if saveEval or saveTrain:
            plot_evals("name_eval", eval_params, rewards, stds, history, 
                       savePath=savePath)

    if saveModel:
        model.save(os.path.join(savePath, name))
    return rewards, stds, histories