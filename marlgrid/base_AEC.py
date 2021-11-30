# Multi-agent gridworld.
# Based on MiniGrid: https://github.com/maximecb/gym-minigrid.

#workign in emcog-2

import gym
import numpy as np
import gym_minigrid
from enum import IntEnum
import math
import warnings

from .objects import WorldObj, Wall, Goal, Lava, GridAgent, BonusTile, BulkObj, COLORS
from .agents import GridAgentInterface
from .rendering import SimpleImageViewer
from gym_minigrid.rendering import fill_coords, point_in_rect, downsample, highlight_img
from gym.spaces import Discrete, Box
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers

TILE_PIXELS = 32
NUM_ITERS = 100
NONE = 4

class ObjectRegistry:
    '''
    This class contains dicts that map objects to numeric keys and vise versa.
    Used so that grid worlds can represent objects using numerical arrays rather 
        than lists of lists of generic objects.
    '''
    def __init__(self, objs=[], max_num_objects=1000):
        self.key_to_obj_map = {}
        self.obj_to_key_map = {}
        self.max_num_objects = max_num_objects
        for obj in objs:
            self.add_object(obj)

    def get_next_key(self):
        for k in range(self.max_num_objects):
            if k not in self.key_to_obj_map:
                break
        else:
            raise ValueError("Object registry full.")
        return k

    def __len__(self):
        return len(self.id_to_obj_map)

    def add_object(self, obj):
        new_key = self.get_next_key()
        self.key_to_obj_map[new_key] = obj
        self.obj_to_key_map[obj] = new_key
        return new_key

    def contains_object(self, obj):
        return obj in self.obj_to_key_map

    def contains_key(self, key):
        return key in self.key_to_obj_map

    def get_key(self, obj):
        if obj in self.obj_to_key_map:
            return self.obj_to_key_map[obj]
        else:
            return self.add_object(obj)

    # 5/4/2020 This gets called A LOT. Replaced calls to this function with direct dict gets
    #           in an attempt to speed things up. Probably didn't make a big difference.
    def obj_of_key(self, key):
        return self.key_to_obj_map[key]


def rotate_grid(grid, rot_k):
    '''
    This function basically replicates np.rot90 (with the correct args for rotating images).
    But it's faster.
    '''
    rot_k = rot_k % 4
    if rot_k==3:
        return np.moveaxis(grid[:,::-1], 0, 1)
    elif rot_k==1:
        return np.moveaxis(grid[::-1,:], 0, 1)
    elif rot_k==2:
        return grid[::-1,::-1]
    else:
        return grid


def MultiGridEnv():
    #TODO: Check this is used?
    '''
    The env function wraps the environment in 3 wrappers by default. These
    wrappers contain logic that is common to many pettingzoo environments.
    We recommend you use at least the OrderEnforcingWrapper on your own environment
    to provide sane error messages. You can find full documentation for these methods
    elsewhere in the developer documentation.
    '''
    env = raw_MultiGridEnv()
    #env = wrappers.CaptureStdoutWrapper(env)
    #env = wrappers.AssertOutOfBoundsWrapper(env)
    #env = wrappers.OrderEnforcingWrapper(env)
    return env

class MultiGrid:

    tile_cache = {}

    def __init__(self, shape=(11,11), obj_reg=None, orientation=0):
        self.orientation = orientation
        if isinstance(shape, tuple):
            self.width, self.height = shape
            self.grid = np.zeros((self.width, self.height), dtype=np.uint8)  # w,h
        elif isinstance(shape, np.ndarray):
            self.width, self.height = shape.shape
            self.grid = shape
        else:
            raise ValueError("Must create grid from shape tuple or array.")

        if self.width < 3 or self.height < 3:
            raise ValueError("Grid needs width, height >= 3")

        self.obj_reg = ObjectRegistry(objs=[None]) if obj_reg is None else obj_reg

    @property
    def opacity(self):
        transparent_fun = np.vectorize(lambda k: (self.obj_reg.key_to_obj_map[k].see_behind() if hasattr(self.obj_reg.key_to_obj_map[k], 'see_behind') else True))
        return ~transparent_fun(self.grid)

    def __getitem__(self, *args, **kwargs):
        return self.__class__(
            np.ndarray.__getitem__(self.grid, *args, **kwargs),
            obj_reg=self.obj_reg,
            orientation=self.orientation,
        )

    def rotate_left(self, k=1):
        return self.__class__(
            rotate_grid(self.grid, rot_k=k), # np.rot90(self.grid, k=k),
            obj_reg=self.obj_reg,
            orientation=(self.orientation - k) % 4,
        )


    def slice(self, topX, topY, width, height, rot_k=0):
        """
        Get a subset of the grid
        """
        sub_grid = self.__class__(
            (width, height),
            obj_reg=self.obj_reg,
            orientation=(self.orientation - rot_k) % 4,
        )
        x_min = max(0, topX)
        x_max = min(topX + width, self.width)
        y_min = max(0, topY)
        y_max = min(topY + height, self.height)

        x_offset = x_min - topX
        y_offset = y_min - topY
        sub_grid.grid[
            x_offset : x_max - x_min + x_offset, y_offset : y_max - y_min + y_offset
        ] = self.grid[x_min:x_max, y_min:y_max]

        sub_grid.grid = rotate_grid(sub_grid.grid, rot_k)

        sub_grid.width, sub_grid.height = sub_grid.grid.shape

        return sub_grid

    def set(self, i, j, obj):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        self.grid[i, j] = self.obj_reg.get_key(obj)

    def get(self, i, j):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height

        return self.obj_reg.key_to_obj_map[self.grid[i, j]]

    def horz_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x + i, y, obj_type())

    def vert_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, obj_type())

    def wall_rect(self, x, y, w, h, obj_type=Wall):
        self.horz_wall(x, y, w, obj_type=obj_type)
        self.horz_wall(x, y + h - 1, w, obj_type=obj_type)
        self.vert_wall(x, y, h, obj_type=obj_type)
        self.vert_wall(x + w - 1, y, h, obj_type=obj_type)

    def __str__(self):
        render = (
            lambda x: "  "
            if x is None or not hasattr(x, "str_render")
            else x.str_render(dir=self.orientation)
        )
        hstars = "*" * (2 * self.width + 2)
        return (
            hstars
            + "\n"
            + "\n".join(
                "*" + "".join(render(self.get(i, j)) for i in range(self.width)) + "*"
                for j in range(self.height)
            )
            + "\n"
            + hstars
        )

    def encode(self, vis_mask=None):
        """
        Produce a compact numpy encoding of the grid
        """

        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros((self.width, self.height, 3), dtype="uint8")

        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)
                    if v is None:
                        array[i, j, :] = 0
                    else:
                        array[i, j, :] = v.encode()
        return array

    @classmethod
    def decode(cls, array):
        raise NotImplementedError
        width, height, channels = array.shape
        assert channels == 3
        vis_mask[i, j] = np.ones(shape=(width, height), dtype=np.bool)
        grid = cls((width, height))

    
    @classmethod
    def cache_render_fun(cls, key, f, *args, **kwargs):
        if key not in cls.tile_cache:
            cls.tile_cache[key] = f(*args, **kwargs)
        return np.copy(cls.tile_cache[key])

    @classmethod
    def cache_render_obj(cls, obj, tile_size, subdivs):
        if obj is None:
            return cls.cache_render_fun((tile_size, None), cls.empty_tile, tile_size, subdivs)
        else:
            img = cls.cache_render_fun(
                (tile_size, obj.__class__.__name__, *obj.encode()),
                cls.render_object, obj, tile_size, subdivs
            )
            if hasattr(obj, 'render_post'):
                return obj.render_post(img)
            else:
                return img

    @classmethod
    def empty_tile(cls, tile_size, subdivs):
        alpha = max(0, min(20, tile_size-10))
        img = np.full((tile_size, tile_size, 3), alpha, dtype=np.uint8)
        img[1:,:-1] = 0
        return img

    @classmethod
    def render_object(cls, obj, tile_size, subdivs):
        img = np.zeros((tile_size*subdivs,tile_size*subdivs, 3), dtype=np.uint8)
        obj.render(img)
        # if 'Agent' not in obj.type and len(obj.agents) > 0:
        #     obj.agents[0].render(img)
        return downsample(img, subdivs).astype(np.uint8)

    @classmethod
    def blend_tiles(cls, img1, img2):
        '''
        This function renders one "tile" on top of another. Kinda janky, works surprisingly well.
        Assumes img2 is a downscaled monochromatic with a black (0,0,0) background.
        '''
        alpha = img2.sum(2, keepdims=True)
        max_alpha = alpha.max()
        if max_alpha == 0:
            return img1
        return (
            ((img1 * (max_alpha-alpha))+(img2*alpha)
            )/max_alpha
        ).astype(img1.dtype)

    @classmethod
    def render_tile(cls, obj, tile_size=TILE_PIXELS, subdivs=3, top_agent=None):
        subdivs = 3

        if obj is None:
            img = cls.cache_render_obj(obj, tile_size, subdivs)
        else:
            if ('Agent' in obj.type) and (top_agent in obj.agents):
                # If the tile is a stack of agents that includes the top agent, then just render the top agent.
                img = cls.cache_render_obj(top_agent, tile_size, subdivs)
            else: 
                # Otherwise, render (+ downsize) the item in the tile.
                img = cls.cache_render_obj(obj, tile_size, subdivs)
                # If the base obj isn't an agent but has agents on top, render an agent and blend it in.
                if len(obj.agents)>0 and 'Agent' not in obj.type:
                    if top_agent in obj.agents:
                        img_agent = cls.cache_render_obj(top_agent, tile_size, subdivs)
                    else:
                        img_agent = cls.cache_render_obj(obj.agents[0], tile_size, subdivs)
                    img = cls.blend_tiles(img, img_agent)

            # Render the tile border if any of the corners are black.
            if (img[([0,0,-1,-1],[0,-1,0,-1])]==0).all(axis=-1).any():
                img = img + cls.cache_render_fun((tile_size, None), cls.empty_tile, tile_size, subdivs)
        return img

    def render(self, tile_size, highlight_mask=None, visible_mask=None, top_agent=None):
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px), dtype=np.uint8)[...,None]+COLORS['shadow']

        for j in range(0, self.height):
            for i in range(0, self.width):
                if visible_mask is not None and not visible_mask[i,j]:
                    continue
                obj = self.get(i, j)

                tile_img = MultiGrid.render_tile(
                    obj,
                    tile_size=tile_size,
                    top_agent=top_agent
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size

                img[ymin:ymax, xmin:xmax, :] = rotate_grid(tile_img, self.orientation)
        
        if highlight_mask is not None:
            hm = np.kron(highlight_mask.T, np.full((tile_size, tile_size), 255, dtype=np.uint16)
                )[...,None] # arcane magic.
            img = np.right_shift(img.astype(np.uint16)*8+hm*2, 3).clip(0,255).astype(np.uint8)

        img = img.astype(np.uint8)
        return img

def unused_raw_MultiGridEnv():
    '''
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    '''
    env = para_MultiGridEnv()
    env = from_parallel(env)
    return env

class para_MultiGridEnv(ParallelEnv):
    '''
    The metadata holds environment constants. From gym, we inherit the "render.modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    '''
    metadata = {'render.modes': ['human'], "name": "multigrid_alpha"}

    def __init__(
        self,
        agents = [],
        grid_size=None,
        width=11,
        height=11,
        max_steps=100,
        reward_decay=True,
        seed=1337,
        respawn=False,
        ghost_mode=True,
        agent_spawn_kwargs = {}
    ):
        '''
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        '''
        if grid_size is not None:
            assert width == None and height == None
            width, height = grid_size, grid_size

        self.respawn = respawn

        self.window = None

        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.reward_decay = reward_decay
        self.seed(seed=seed)
        self.agent_spawn_kwargs = agent_spawn_kwargs
        self.ghost_mode = ghost_mode
        self.agent_view_size=45

        self.agents = agents
        self.grid = MultiGrid(shape=(width,height)) #added this, not sure where grid comes from in og

        self.possible_agents = ["player_" + str(r) for r in range(len(agents))]
        #self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))

        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        self.action_spaces = {agent: Discrete(5) for agent in self.possible_agents}
        
        
        self.step_count = 0
        #obshape = (45,45,3)
        
        self.observation_spaces = {agent: Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, 3),
            dtype='uint8'
        ) for agent in self.possible_agents}
        '''self.observation_space = Dict({
            'image': self.observation_space
        })'''
        self.agent_instances = {agent for agent in agents}

        self.instance_from_name = {name: agent for name, agent in zip(self.possible_agents, agents)}


    '''def action_space(self, name):
        return self.action_spaces[name]

    def observation_space(self, name):
        return self.observation_spaces[name]'''

    def seed(self, seed=1337):
        # Seed the random number generator
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        return [seed]

    def check_agent_position_integrity(self, title=''):
        '''
        This function checks whether each agent is present in the grid in exactly one place.
        This is particularly helpful for validating the world state when ghost_mode=False and
        agents can stack, since the logic for moving them around gets a bit messy.
        Prints a message and drops into pdb if there's an inconsistency.
        '''
        agent_locs = [[] for _ in range(len(self.agents))]
        for i in range(self.grid.width):
            for j in range(self.grid.height):
                x = self.grid.get(i,j)
                for k,agent in enumerate(self.agents):
                    if x==agent:
                        agent_locs[k].append(('top', (i,j)))
                    if hasattr(x, 'agents') and agent in x.agents:
                        agent_locs[k].append(('stacked', (i,j)))
        if not all([len(x)==1 for x in agent_locs]):
            print(f"{title} > Failed integrity test!")
            for a, al in zip(self.agents, agent_locs):
                print(" > ", a.color,'-', al)
            import pdb; pdb.set_trace()

    def observe(self, agent):
        '''
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        '''
        # observation of one agent is the previous state of the other
        return np.array(self.observations[agent])

    def close(self):
        '''
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        '''
        pass

    def reset(self):
        '''
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - dones
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.

        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        '''
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: NONE for agent in self.agents}
        self.observations = {agent: self.gen_agent_obs(a) for agent, a in zip(self.agents, self.agent_instances)}
        self.num_moves = 0
        '''
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        '''
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        for agent in self.agent_instances:
            agent.agents = []
            agent.reset(new_episode=True)

        self._gen_grid(self.width, self.height)

        for agent in self.agent_instances:
            if agent.spawn_delay == 0:
                self.place_obj(agent, **self.agent_spawn_kwargs)
                agent.activate()

        return self.observations

    def step(self, actions):

        '''
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - dones
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        '''

        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}

        #temp fix, TODO: check if done properly
        env_done = False

        #print('acts', actions)
        # Spawn agents if it's time.
        if self.dones[self.agent_selection]:
            # handles stepping an agent which is already done
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next done agent,  or if there are no more done agents, to the next live agent
            
            #changed this from returning self._was_done_step(action)
            #self.agent_selection = self._agent_selector.next()
            return self.observations, self.rewards, self.dones, {}

        agent_name = self.agent_selection
        agent = self.instance_from_name[agent_name]

        if not agent.active and not agent.done and self.step_count >= agent.spawn_delay:
            self.place_obj(agent, **self.agent_spawn_kwargs)
            agent.activate()

        self._cumulative_rewards[agent] = 0
        #self.state[self.agent_selection] = action

        for agent_name in actions:
            action = actions[agent_name]
            agent = self.instance_from_name[agent_name]
            #agent_no, (agent, action) = iter_agents[shuffled_ix]
            agent.step_reward = 0

            if agent.active:

                cur_pos = agent.pos[:]
                cur_cell = self.grid.get(*cur_pos)
                #print('cell:' , cur_cell)
                fwd_pos = agent.front_pos[:]
                fwd_cell = self.grid.get(*fwd_pos)
                agent_moved = False

                # Rotate left
                if action == agent.actions.left:
                    agent.dir = (agent.dir - 1) % 4

                # Rotate right
                elif action == agent.actions.right:
                    agent.dir = (agent.dir + 1) % 4

                # Move forward
                elif action == agent.actions.forward:
                    # Under the follow conditions, the agent can move forward.
                    can_move = fwd_cell is None or fwd_cell.can_overlap()
                    if self.ghost_mode is False and isinstance(fwd_cell, GridAgent):
                        can_move = False

                    if can_move:
                        agent_moved = True
                        # Add agent to new cell
                        if fwd_cell is None:
                            self.grid.set(*fwd_pos, agent)
                            agent.pos = fwd_pos
                        else:
                            fwd_cell.agents.append(agent)
                            agent.pos = fwd_pos

                        # Remove agent from old cell
                        if cur_cell == agent:
                            self.grid.set(*cur_pos, None)
                        elif cur_cell != None:
                            # used to just be else... possibly issue because not spawning agents correctly
                            assert cur_cell.can_overlap()
                            #also this if used to not be here
                            if agent in cur_cell.agents:
                                cur_cell.agents.remove(agent)

                        # Add agent's agents to old cell
                        for left_behind in agent.agents:
                            cur_obj = self.grid.get(*cur_pos)
                            if cur_obj is None:
                                self.grid.set(*cur_pos, left_behind)
                            elif cur_obj.can_overlap():
                                cur_obj.agents.append(left_behind)
                            else: # How was "agent" there in teh first place?
                                raise ValueError("?!?!?!")

                        # After moving, the agent shouldn't contain any other agents.
                        agent.agents = [] 
                        # test_integrity(f"After moving {agent.color} fellow")

                        # Rewards can be got iff. fwd_cell has a "get_reward" method
                        if hasattr(fwd_cell, 'get_reward'):
                            rwd = fwd_cell.get_reward(agent)
                            if bool(self.reward_decay):
                                rwd *= (1.0-0.9*(self.step_count/self.max_steps))
                            
                            # removed, unclear what for
                            #step_rewards[agent_no] += rwd
                            
                            agent.reward(rwd)
                            

                        if isinstance(fwd_cell, (Lava, Goal)):
                            agent.done = True
                            #added below
                            print('great success!')
                            self.dones[agent_name] = True

                # TODO: verify pickup/drop/toggle logic in an environment that 
                #  supports the relevant interactions.
                # Pick up an object
                elif action == agent.actions.pickup:
                    if fwd_cell and fwd_cell.can_pickup():
                        if agent.carrying is None:
                            agent.carrying = fwd_cell
                            agent.carrying.cur_pos = np.array([-1, -1])
                            self.grid.set(*fwd_pos, None)
                    else:
                        pass

                # Drop an object
                elif action == agent.actions.drop:
                    if not fwd_cell and agent.carrying:
                        self.grid.set(*fwd_pos, agent.carrying)
                        agent.carrying.cur_pos = fwd_pos
                        agent.carrying = None
                    else:
                        pass

                # Toggle/activate an object
                elif action == agent.actions.toggle:
                    if fwd_cell:
                        wasted = bool(fwd_cell.toggle(agent, fwd_pos))
                    else:
                        pass

                # Done action (not used by default)
                elif action == agent.actions.done:
                    pass

                else:
                    raise ValueError(f"Environment can't handle action {action}.")

                agent.on_step(fwd_cell if agent_moved else None)

        
        # If any of the agents individually are "done" (hit lava or in some cases a goal) 
        #   but the env requires respawning, then respawn those agents.
        for agent in self.agent_instances:
            if agent.done:
                if self.respawn:
                    resting_place_obj = self.grid.get(*agent.pos)
                    if resting_place_obj == agent:
                        if agent.agents:
                            self.grid.set(*agent.pos, agent.agents[0])
                            agent.agents[0].agents += agent.agents[1:]
                        else:
                            self.grid.set(*agent.pos, None)
                    else:
                        resting_place_obj.agents.remove(agent)
                        resting_place_obj.agents += agent.agents[:]
                        agent.agents = []
                        
                    agent.reset(new_episode=False)
                    self.place_obj(agent, **self.agent_spawn_kwargs)
                    agent.activate()
                else: # if the agent shouldn't be respawned, then deactivate it.
                    agent.deactivate()

        # The episode overall is done if all the agents are done, or if it exceeds the step limit.
        #done = (self.step_count >= self.max_steps) or all([agent.done for agent in self.agents])

        #self.observations[agentname] = self.gen_agent_obs(agent)

        # made true since we update all agents
        if True or self._agent_selector.is_last():
            # rewards for all agents are placed in the .rewards dictionary
            self.rewards[agent] = 0 #reward

            self.num_moves += 1
            self.step_count += 1
            # The dones dictionary must be updated for all players.
            self.dones = {agent: self.num_moves >= NUM_ITERS for agent in self.agents}

            # observe the current state
            for agentName, agent in zip(self.agents, self.agent_instances):
                self.observations[agentName] = self.gen_agent_obs(agent)
        else:
            # necessary so that observe() returns a reasonable observation at all times.
            self.state[self.agents[0]] = NONE #todo expand this
            # no rewards are allocated until both players give an action
            
            rewards = {}
            #self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards

        dones = {agent: env_done for agent in self.agents}

        # current observation is just the other player's most recent action
        observations = {self.agents[i]: self.gen_agent_obs(self.instance_from_name[self.agents[i]]) for i in range(len(self.agents))} #currently 0
        rewards = {agent: 0 for agent in self.agents}
        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent: {} for agent in self.agents}

        #self._accumulate_rewards()

        return observations, rewards, dones, infos

    def gen_obs_grid(self, agent):
        # If the agent is inactive, return an empty grid and a visibility mask that hides everything.
        if not agent.active:
            # below, not sure orientation is correct but as of 6/27/2020 that doesn't matter because
            # agent views are usually square and this grid won't be used for anything.
            grid = MultiGrid((agent.view_size, agent.view_size), orientation=agent.dir+1)
            vis_mask = np.zeros((agent.view_size, agent.view_size), dtype=np.bool)
            return grid, vis_mask

        topX, topY, botX, botY = agent.get_view_exts()

        grid = self.grid.slice(
            topX, topY, agent.view_size, agent.view_size, rot_k=agent.dir + 1
        )

        # Process occluders and visibility
        # Note that this incurs some slight performance cost
        vis_mask = agent.process_vis(grid.opacity)

        # Warning about the rest of the function:
        #  Allows masking away objects that the agent isn't supposed to see.
        #  But breaks consistency between the states of the grid objects in the parial views
        #   and the grid objects overall.
        if len(getattr(agent, 'hide_item_types', []))>0:
            for i in range(grid.width):
                for j in range(grid.height):
                    item = grid.get(i,j)
                    if (item is not None) and (item is not agent) and (item.type in agent.hide_item_types):
                        if len(item.agents) > 0:
                            grid.set(i,j,item.agents[0])
                        else:
                            grid.set(i,j,None)

        return grid, vis_mask
    def gen_agent_obs(self, agent):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """
        # issue: should return uint8 observations, not int64
        grid, vis_mask = self.gen_obs_grid(agent)
        grid_image = grid.render(tile_size=agent.view_tile_size, visible_mask=vis_mask, top_agent=agent)
        if agent.observation_style=='image':
            return grid_image
        else:
            ret = {'pov': grid_image}
            if agent.observe_rewards:
                ret['reward'] = getattr(agent, 'step_reward', 0)
            if agent.observe_position:
                agent_pos = agent.pos if agent.pos is not None else (0,0)
                ret['position'] = np.array(agent_pos)/np.array([self.width, self.height], dtype=np.float)
            if agent.observe_orientation:
                agent_dir = agent.dir if agent.dir is not None else 0
                ret['orientation'] = agent_dir
            return ret

    def gen_obs(self):
        return [self.gen_agent_obs(agent) for agent in self.agent_instances]

    def __str__(self):
        return self.grid.__str__()

    def put_obj(self, obj, i, j):
        """
        Put an object at a specific position in the grid. Replace anything that is already there.
        """
        self.grid.set(i, j, obj)
        if obj is not None:
            obj.set_position((i,j))
        return True

    def try_place_obj(self,obj, pos):
        ''' Try to place an object at a certain position in the grid.
        If it is possible, then do so and return True.
        Otherwise do nothing and return False. '''
        # grid_obj: whatever object is already at pos.
        grid_obj = self.grid.get(*pos)

        # If the target position is empty, then the object can always be placed.
        if grid_obj is None:
            self.grid.set(*pos, obj)
            obj.set_position(pos)
            return True

        # Otherwise only agents can be placed, and only if the target position can_overlap.
        if not (grid_obj.can_overlap() and obj.is_agent):
            return False

        # If ghost mode is off and there's already an agent at the target cell, the agent can't
        #   be placed there.
        if (not self.ghost_mode) and (grid_obj.is_agent or (len(grid_obj.agents)>0)):
            return False

        grid_obj.agents.append(obj)
        obj.set_position(pos)
        return True

    def place_obj(self, obj, top=(0,0), size=None, reject_fn=None, max_tries=1e5):
        max_tries = int(max(1, min(max_tries, 1e5)))
        top = (max(top[0], 0), max(top[1], 0))
        if size is None:
            size = (self.grid.width, self.grid.height)
        bottom = (min(top[0] + size[0], self.grid.width), min(top[1] + size[1], self.grid.height))

        # agent_positions = [tuple(agent.pos) if agent.pos is not None else None for agent in self.agents]
        for try_no in range(max_tries):
            pos = self.np_random.randint(top, bottom)
            if (reject_fn is not None) and reject_fn(pos):
                continue
            else:
                if self.try_place_obj(obj, pos):
                    break
        else:
            raise RecursionError("Rejection sampling failed in place_obj.")

        return pos

    def place_agents(self, top=None, size=None, rand_dir=True, max_tries=1000):
        # warnings.warn("Placing agents with the function place_agents is deprecated.")
        pass


    def render(
        self,
        mode="human",
        close=False,
        highlight=True,
        tile_size=TILE_PIXELS,
        show_agent_views=True,
        max_agents_per_col=3,
        agent_col_width_frac = 0.3,
        agent_col_padding_px = 2,
        pad_grey = 100
    ):
        """
        Render the whole-grid human view
        """

        if close:
            if self.window:
                self.window.close()
            return

        if False and mode == "human" and not self.window:
            # from gym.envs.classic_control.rendering import SimpleImageViewer

            self.window = SimpleImageViewer(caption="Marlgrid")
        #print('test')
        # Compute which cells are visible to the agent
        highlight_mask = np.full((self.width, self.height), False, dtype=np.bool)
        #print('testuy')
        for agentname, agent in zip(self.agents, self.agent_instances):
            #print('agent', agent, 'an', agentname, 'a', agent[agentname])
            if agent.active:
                #print('active')
                xlow, ylow, xhigh, yhigh = agent.get_view_exts()
                dxlow, dylow = max(0, 0-xlow), max(0, 0-ylow)
                dxhigh, dyhigh = max(0, xhigh-self.grid.width), max(0, yhigh-self.grid.height)
                if agent.see_through_walls:
                    highlight_mask[xlow+dxlow:xhigh-dxhigh, ylow+dylow:yhigh-dyhigh] = True
                else:
                    a,b = self.gen_obs_grid(agent)
                    highlight_mask[xlow+dxlow:xhigh-dxhigh, ylow+dylow:yhigh-dyhigh] |= (
                        rotate_grid(b, a.orientation)[dxlow:(xhigh-xlow)-dxhigh, dylow:(yhigh-ylow)-dyhigh]
                    )
        #print('test2')
        #print('h', highlight) #true
        # Render the whole grid
        img = self.grid.render(
            tile_size, highlight_mask=highlight_mask if highlight else None
        )
        rescale = lambda X, rescale_factor=2: np.kron(
            X, np.ones((int(rescale_factor), int(rescale_factor), 1))
        )
        #print('test3', img)

        if show_agent_views: 
            #print('agent_views')

            target_partial_width = int(img.shape[0]*agent_col_width_frac-2*agent_col_padding_px)
            target_partial_height = (img.shape[1]-2*agent_col_padding_px)//max_agents_per_col

            agent_views = [self.gen_agent_obs(agent) for agent in self.agent_instances]
            agent_views = [view['pov'] if isinstance(view, dict) else view for view in agent_views]
            agent_views = [rescale(view, min(target_partial_width/view.shape[0], target_partial_height/view.shape[1])) for view in agent_views]
            # import pdb; pdb.set_trace()
            agent_views = [agent_views[pos:pos+max_agents_per_col] for pos in range(0, len(agent_views), max_agents_per_col)]

            f_offset = lambda view: np.array([target_partial_height - view.shape[1], target_partial_width - view.shape[0]])//2
            
            cols = []
            for col_views in agent_views:
                col = np.full(( img.shape[0],target_partial_width+2*agent_col_padding_px,3), pad_grey, dtype=np.uint8)
                for k, view in enumerate(col_views):
                    offset = f_offset(view) + agent_col_padding_px
                    offset[0] += k*target_partial_height
                    col[offset[0]:offset[0]+view.shape[0], offset[1]:offset[1]+view.shape[1],:] = view
                cols.append(col)

            img = np.concatenate((img, *cols), axis=1)
        
        
        '''if mode == "human":
            if not self.window.isopen:
                self.window.imshow(img)
                self.window.window.set_caption("Marlgrid")
            else:
                self.window.imshow(img)'''
        
        #print('returning', img)
        return img