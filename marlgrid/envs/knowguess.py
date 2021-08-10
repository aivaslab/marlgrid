from ..base import MultiGridEnv, MultiGrid
from ..objects import *


class KnowGuessEnv(MultiGridEnv):
    """
    """

    mission = "use the key to open the door and then get to the goal"
    metadata = {}

    apos = [(7,3,1),(7,11,3)]

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

class KnowGuessEnv2(MultiGridEnv):
    """
    RL version of above
    """

    mission = "use the key to open the door and then get to the goal"
    metadata = {}
    
    apos = [(4,3,1),(4,11,3), (11, 7,2)]

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = MultiGrid((width, height))

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        for x in range(1,9):
            self.put_obj(Wall(color="blue"), x, height//3)
            self.put_obj(Wall(color="blue"), x, 2*height//3-1)

        #for y in range(1,height-1):
        #    self.put_obj(Box(color="orange"), 10, y)

        for y in range(6,8+1):
            self.put_obj(Wall(), 8, y)


        self.put_obj(Ball(color="green"), 4, height//2)
        self.put_obj(Goal(color="green", reward=1), 2, height//2)
        self.put_obj(Goal(color="green", reward=1), 6, height//2)


class KnowGuessEnv3(MultiGridEnv):
    """
    Visor
    """

    mission = "use the key to open the door and then get to the goal"
    metadata = {}
    apos = [(4,3,1),(4,11,3), (11, 7,2)]

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = MultiGrid((width, height))

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        for x in range(1,9):
            self.put_obj(Wall(color="blue"), x, height//3)
            self.put_obj(Goal(color="orange", reward=0), x, 2*height//3-1)

        #for y in range(1,height-1):
        #    self.put_obj(Box(color="orange"), 10, y)

        for y in range(6,8+1):
            self.put_obj(Wall(), 8, y)


        self.put_obj(Ball(color="green"), 4, height//2)
        self.put_obj(Goal(color="green", reward=1), 2, height//2)
        self.put_obj(Goal(color="green", reward=1), 6, height//2)



        
