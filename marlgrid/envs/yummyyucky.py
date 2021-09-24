from ..base import MultiGridEnv, MultiGrid
from ..objects import *
import random
import math

class YummyYuckyEnv0(MultiGridEnv):
    """
    """

    mission = "yummy yucky simple: go to the correct color, of 2."
    metadata = {}

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = MultiGrid((width, height))

        chosen = 0 # choose green as the good color
        mirror1 = random.choice([-1,1])

        c = ['green', 'blue']

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        for x in range(2):
            r = 1 if x == chosen else -1
            self.put_obj(Goal(color=c[x], reward=r), width//2 + 1*(x*2-1)*mirror1, height//2)


        self.agent_spawn_kwargs = {"top":(1,1)}
        self.place_agents(**self.agent_spawn_kwargs)


class YummyYuckyEnv1(MultiGridEnv):
    """
    """

    mission = "yummy yucky"
    metadata = {}

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = MultiGrid((width, height))

        chosen = 0#random.choice([0,1])
        mirror1 = random.choice([-1,1])
        mirror2 = random.choice([-1,1])

        c = ['green', 'blue']

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        for x in range(2):
            r = 1 if x == chosen else -1
            self.put_obj(Goal(color=c[x], reward=r), width//2 + 3*(x*2-1)*mirror1, height//2)

        for x in range(2):
            r = 1 if x == chosen else -1
            self.put_obj(Goal(color=c[x], reward=r), width//2 + 3*(x*2-1), height//2-3*(x*2-1)*mirror2)
            self.put_obj(Goal(color=c[not x], reward=r), width//2 + 3*(x*2-1), height//2+3*(x*2-1)*mirror2)

        self.agent_spawn_kwargs = {"top":(1,1)}
        self.place_agents(**self.agent_spawn_kwargs)


class YummyYuckyEnv3(MultiGridEnv):
    """
    """

    mission = "yummy yucky"
    metadata = {}

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = MultiGrid((width, height))

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        for x in range(4):
            r = 1 if x == chosen else -1
            self.put_obj(Goal(color=c[x], reward=r), width//2 + int(3*math.cos(x*3.14/2)), height//2 + int(3*math.sin(x*3.14/2)))

        self.agent_spawn_kwargs = {"color":"green", "view_offset": 0}
        self.place_agents(**self.agent_spawn_kwargs)
