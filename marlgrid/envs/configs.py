def get_config(name):
    '''
    returns env and then player config
    '''
    if name == 'cc':
        return {
        "env_class": "ClutteredGoalCycleEnv",
        "grid_size": 13,
        "max_steps": 250,
        "clutter_density": 0.15,
        "respawn": True,
        "ghost_mode": True,
        "reward_decay": False,
        "n_bonus_tiles": 3,
        "initial_reward": True,
        "penalty": -1.5
        }, {
        "view_size": 7,
        "view_offset": 3,
        "view_tile_size": 3,
        "observation_style": "image",
        "see_through_walls": False,
        "color": "prestige"
        }
    elif name == '0':
        return {
        "env_class": "ClutteredGoalCycleEnv",
        "grid_size": 13,
        "max_steps": 250,
        "clutter_density": 0.15,
        "respawn": True,
        "ghost_mode": True,
        "reward_decay": False,
        "n_bonus_tiles": 3,
        "initial_reward": True,
        "penalty": -1.5
        }, {
        "view_size": 7,
        "view_offset": 3,
        "view_tile_size": 3,
        "observation_style": "image",
        "see_through_walls": False,
        "color": "prestige"
        }
    elif name == 'd':
        return {
        "env_class": "DoorKeyEnv",
        "grid_size": 8,
        "max_steps": 250,
        "respawn": True,
        "ghost_mode": True,
        "reward_decay": True,
        }, {
        "view_size": 7,
        "view_offset": 3,
        "view_tile_size": 3,
        "observation_style": "image",
        "see_through_walls": False,
        "color": "prestige"
        }
    elif name == 'cf':
        return {
        "env_class": "ContentFBEnv",
        "grid_size": 15,
        "max_steps": 250,
        "respawn": True,
        "ghost_mode": True,
        "reward_decay": True,
        }, {
        "view_size": 7,
        "view_offset": 3,
        "view_tile_size": 3,
        "observation_style": "image",
        "see_through_walls": False,
        "color": "prestige"
        }
    elif name == 'yy0':
        return {
        "env_class": "YummyYuckyEnv0",
        "grid_size": 9,
        "max_steps": 100,
        "respawn": True,
        "ghost_mode": True,
        "reward_decay": True,
        }, {
        "view_size": 7,
        "view_offset": 3,
        "view_tile_size": 3,
        "observation_style": "image",
        "see_through_walls": False,
        "color": "prestige"
        }
    elif name == 'yy1':
        return {
        "env_class": "YummyYuckyEnv1",
        "grid_size": 7,
        "max_steps": 100,
        "respawn": True,
        "ghost_mode": True,
        "reward_decay": True,
        }, {
        "view_size": 5,
        "view_offset": 2,
        "view_tile_size": 3,
        "observation_style": "image",
        "see_through_walls": False,
        "color": "prestige"
        }
    elif name == '0':
        return {
        "env_class": "ClutteredGoalCycleEnv",
        "grid_size": 13,
        "max_steps": 250,
        "clutter_density": 0.15,
        "respawn": True,
        "ghost_mode": True,
        "reward_decay": False,
        "n_bonus_tiles": 3,
        "initial_reward": True,
        "penalty": -1.5
        }, {
        "view_size": 7,
        "view_offset": 3,
        "view_tile_size": 3,
        "observation_style": "image",
        "see_through_walls": False,
        "color": "prestige"
        }