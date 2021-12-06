from ..base_AEC import *
from ..objects import *
from random import randrange
import random
import math

class ContentFBEnv(para_MultiGridEnv):
    """
    Environment with a door and key, sparse reward.
    Similar to DoorKeyEnv in 
        https://github.com/maximecb/gym-minigrid/blob/master/gym_minigrid/envs/doorkey.py
    """

    mission = "use the key to open the door and then get to the goal"
    metadata = {}
    

    def init_agents(self, arg, agent_kwargs):
        if arg == 0:
            self.apos = [(6,11,3)]
        for agent in self.apos:
            self.add_agent(GridAgentInterface(**agent_kwargs))

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = MultiGrid((width, height))
        colors = random.sample(['green','purple','orange','yellow','blue','pink','red'], 4)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width-2, height)

        for k, x in enumerate(range(0,width-4,4)):
            self.grid.wall_rect(x, 0, 5, 5)
            self.put_obj(Goal(color=colors[k], reward=1), x+2, 2)
            self.put_obj(Door(color=colors[k]), x+2, 4)
            #self.put_obj(Key(color=colors[k],), x+2, 4)

        self.agent_spawn_kwargs = {'top':(1,1)}
        self.place_agents(**self.agent_spawn_kwargs)

class ContentFBEnv2(para_MultiGridEnv):
    """
    Environment with a door and key, sparse reward.
    Similar to DoorKeyEnv in 
        https://github.com/maximecb/gym-minigrid/blob/master/gym_minigrid/envs/doorkey.py
    """

    mission = "use the key to open the door and then get to the goal"
    metadata = {}
    
    def init_agents(self, arg, agent_kwargs):
        if arg == 0:
            self.apos = [(6,11,3)]
        for agent in self.apos:
            self.add_agent(GridAgentInterface(**agent_kwargs))

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = MultiGrid((width, height))
        colors = ['green','purple','orange']

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width-2, height)

        for k, x in enumerate(range(0,width-4,4)):
            self.grid.wall_rect(x, 0, 5, 5)
            self.put_obj(Ball(color=colors[k],), x+2, 2)
            self.put_obj(Wall(color=colors[(k+1) % 3],), x+2, 4)


        self.agent_spawn_kwargs = {}
        self.place_agents(**self.agent_spawn_kwargs)

