from pettingzoo.utils.conversions import to_parallel
import supersuit as ss

def pz2sb3(env, num_cpus=2):
    '''
    takes an env 
    (wrapper/instantiation function around raw of parallel)
    returns sb3 compatible via supersuit
    '''
    envInstance = to_parallel(env)
    env2 = ss.black_death_v2(ss.pad_action_space_v0(ss.pad_observations_v0(envInstance)))
    env2 = ss.pettingzoo_env_to_vec_env_v1(env2)
    env2 = ss.concat_vec_envs_v1(env2, num_cpus, base_class='stable_baselines3')
    env2.black_death = True
    return env2