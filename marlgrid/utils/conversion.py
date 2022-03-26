from pettingzoo.utils.conversions import aec_to_parallel, parallel_to_aec
import supersuit as ss
from pettingzoo.utils import wrappers

def wrap_env(para_env, **kwargs):
    '''
    The env function wraps the environment in 3 wrappers by default. These
    wrappers contain logic that is common to many pettingzoo environments.
    We recommend you use at least the OrderEnforcingWrapper on your own environment
    to provide sane error messages. You can find full documentation for these methods
    elsewhere in the developer documentation.
    '''
    env = raw_env(para_env, **kwargs)
    env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

def raw_env(para_env, **kwargs):
    '''
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    '''
    env = para_env(**kwargs)
    env = parallel_to_aec(env)
    return env

def pz2sb3(env, num_cpus=2):
    '''
    takes a wrapped env 
    returns sb3 compatible via supersuit concatenated set of vector environments
    '''
    envInstance = aec_to_parallel(env)
    env2 = ss.black_death_v3(ss.pad_action_space_v0(ss.pad_observations_v0(envInstance)))
    env2 = ss.pettingzoo_env_to_vec_env_v1(env2)
    env2 = ss.concat_vec_envs_v1(env2, num_cpus, base_class='stable_baselines3')
    env2.black_death = True
    return env2
