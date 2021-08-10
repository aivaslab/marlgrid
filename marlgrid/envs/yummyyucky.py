from ..base import MultiGridEnv, MultiGrid
from ..objects import *


class YummyYuckyEnv(MultiGridEnv):
    """
    """

    mission = "use the key to open the door and then get to the goal"
    metadata = {}

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = MultiGrid((width, height))

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        for x in range(1,width-1):
            self.put_obj(Wall(color="blue"), x, height//3)
            self.put_obj(Wall(color="blue"), x, 2*height//3-1)

        self.put_obj(Ball(color="green"), width//2, height//2)
        self.put_obj(Goal(color="green", reward=1), width//4, height//2)
        self.put_obj(Goal(color="green", reward=1), 3*width//4, height//2)

        
