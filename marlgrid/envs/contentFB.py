from ..base import MultiGridEnv, MultiGrid
from ..objects import *
from ..agents import GridAgentInterface

class ContentFBEnv(MultiGridEnv):
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
            self.put_obj(Wall(color=colors[k],), x+2, 4)

class ContentFBEnv2(MultiGridEnv):
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

