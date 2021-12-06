from ..base_AEC import *
from ..objects import *
from random import randrange
import random
import math

class CompFeedEnv(para_MultiGridEnv):
    """
    Environment with a door and key, sparse reward.
    Similar to DoorKeyEnv in 
        https://github.com/maximecb/gym-minigrid/blob/master/gym_minigrid/envs/doorkey.py
    """

    mission = "use the key to open the door and then get to the goal"
    metadata = {}

    def init_agents(self, arg, agent_kwargs):
        if arg == 0:
            self.apos = [(2,7,0),(12,7,2)]
        for agent in self.apos:
            self.add_agent(GridAgentInterface(**agent_kwargs))

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = MultiGrid((width, height))

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        self.put_obj(Ball(color="green"), width//2, height//2)
        self.put_obj(Goal(color="green", reward=1), width//2, height//4)
        self.put_obj(Goal(color="green", reward=1), width//2, 3*height//4)

        # Create a vertical splitting wall
        self.grid.vert_wall(3, 0)
        self.grid.vert_wall(width-4, 0)
        self.put_obj(Box(color="orange",), 3, height//2)
        self.put_obj(Box(color="orange",), 3, height//2-1)
        self.put_obj(Box(color="orange",), 3, height//2+1)
        self.put_obj(Box(color="orange",), width-4, height//2-1)
        self.put_obj(Box(color="orange",), width-4, height//2)
        self.put_obj(Box(color="orange",), width-4, height//2+1)

class CompFeedEnv2(para_MultiGridEnv):
    """
    Environment with a door and key, sparse reward.
    Similar to DoorKeyEnv in 
        https://github.com/maximecb/gym-minigrid/blob/master/gym_minigrid/envs/doorkey.py
    """

    mission = "use the key to open the door and then get to the goal"
    metadata = {}

    def init_agents(self, arg, agent_kwargs):
        if arg == 0:
            self.apos = [(2,7,0),(12,5,2),(12,9,2)]
        for agent in self.apos:
            self.add_agent(GridAgentInterface(**agent_kwargs))

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = MultiGrid((width, height))

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        self.put_obj(Ball(color="green"), width//2, height//2)
        self.put_obj(Goal(color="green", reward=1), width//2, height//4)
        self.put_obj(Goal(color="green", reward=1), width//2, 3*height//4)

        # Create a vertical splitting wall
        self.grid.vert_wall(3, 0)
        self.grid.vert_wall(width-4, 0)
        self.put_obj(Box(color="orange",), 3, height//2)
        self.put_obj(Box(color="orange",), 3, height//2-1)
        self.put_obj(Box(color="orange",), 3, height//2+1)

        self.put_obj(Box(color="orange",), width-4, height//4+3)
        self.put_obj(Box(color="orange",), width-4, height//4+1)
        self.put_obj(Box(color="orange",), width-4, height//4+2)

        self.put_obj(Wall(), width-3, height//2)
        self.put_obj(Wall(), width-2, height//2)

        self.put_obj(Box(color="orange",), width-4, 3*height//4-3)
        self.put_obj(Box(color="orange",), width-4, 3*height//4-2)
        self.put_obj(Box(color="orange",), width-4, 3*height//4-1)
        
