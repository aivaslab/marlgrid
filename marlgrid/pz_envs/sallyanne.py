from ..base_AEC import *
from ..objects import *
from random import randrange
import random
import math

class para_SallyAnneEnv(para_MultiGridEnv):
    """
    Environment with a door and key, sparse reward.
    Similar to DoorKeyEnv in 
        https://github.com/maximecb/gym-minigrid/blob/master/gym_minigrid/envs/doorkey.py
    """

    mission = "use the key to open the door and then get to the goal"
    metadata = {'render.modes': ['human', 'rgb_array'], "name": "doorkey"}

    def _rand_int(self, x, y):
        return randrange(x, y)

    def _gen_grid(self, width, height):
        # Create an empty grid
        #print('runnning')
        goals = random.sample([0,0,0,1], 4)
        colors = random.sample(['purple','orange','yellow','blue','pink','red'], 4)

        goal_positions = [(width-2,height//2), (width//2,1), (1,height//2), (width//2, height-2)]
        door_positions = [(width-2 - 1,height//2), (width//2,0 + 2), (0 + 2,height//2), (width//2, height-2 - 1)]
        subgoal_positions = [(width-2 - 2,height//2), (width//2,0 + 3), (0 + 3,height//2), (width//2, height-2 - 2)]

        self.grid = MultiGrid((width, height))

        # Generate the surrounding walls
        
        if random.randrange(50) > 0:
            self.grid.wall_rect(2, 2, width-4, height-4)
        
        if random.randrange(50) > 45:
            self.grid.wall_rect(1, 1, width-2, height-2)
        self.grid.wall_rect(0, 0, width, height)

        # Place doors, goals, subgoals
        for color, goal, goal_p, door, sub in zip(colors, goals, goal_positions, door_positions, subgoal_positions):
            
            self.put_obj(Door(color=color, state=3), door[0], door[1])
            self.place_obj(obj=Key(color), top=(3, 3), size=(width-6, height-6))

            self.put_obj(Goal(color='green', reward=50+50*goal), goal_p[0], goal_p[1])
            if False or goal == 1:
                self.put_obj(SubGoal(color='green'), sub[0], sub[1])

        self.agent_spawn_kwargs = {'top': (2,2), 'size': (width-3, height-3)}
        self.place_agents(**self.agent_spawn_kwargs)
