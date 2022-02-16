from ..base_AEC import *
from ..objects import *
from random import randrange
import random
import math

class para_TutorialEnv(para_MultiGridEnv):
    """
    Environment sparse reward.
    Currently designed for 9x9 envs.
    """

    mission = "get to the goal"
    metadata = {'render.modes': ['human', 'rgb_array'], "name": "tutorial"}
    random_mode = False
    mylevel = 2
    mysublevel = 1
    loading_mode = False
    saving_mode = False
    path = ''

    def _rand_int(self, x, y):
        return randrange(x, y)

    def save_grid(grid, kwargs, path):


    def _gen_grid(self, width, height):
        self.grid = MultiGrid((width, height))
        if self.random_mode:
            level = randrange(1,4)
            if level == 1:
                sublevel = randrange(1,5)
            elif level == 2:
                sublevel = randrange(1,7)
            elif level == 3:
                sublevel = randrange(1,3)
        else:
            level = self.mylevel
            sublevel = self.mysublevel

        colors = random.sample(['purple','orange','yellow','blue','pink','red'], 4)

        #grid and surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        if level == 1: #empty, cluttered, and mazes
            #self.put_obj(Goal(color="green", reward=100), width - 2, height//2)

            if sublevel in [2,3,4]:
                for x in range(2, width-1, 2):
                    for y in range(2, height-1, 2):
                        if sublevel in [3,4]:
                            self.put_obj(Wall(), x, y)
                        else:
                            num = randrange(0,10)
                            if num == 0:
                                self.put_obj(Wall(), x, y)
                            elif num == 1:
                                self.put_obj(Door(color=colors[0], state=randrange(1,4)), x, y)
                            elif num == 2:
                                self.put_obj(Key(color=colors[0], state=randrange(1,4)), x, y)
            if sublevel == 4:
                for i in range(3):
                    num = randrange(0,3)
                    if num == 0:
                        self.place_obj(Wall(), top=(0, 0), size=(width, height))
                    elif num == 1:
                        self.place_obj(Door(color=colors[0], state=randrange(1,4)), top=(0, 0), size=(width, height))
                    elif num == 2:
                        self.place_obj(Key(color=colors[0], state=randrange(1,4)), top=(0, 0), size=(width, height))

            self.place_obj(Goal(color="green", reward=100), top=(0, 0), size=(width, height))


        elif level == 2: #doorkey variants
            splitIdx = self._rand_int(2, width - 2)
            self.grid.vert_wall(splitIdx, 1)

            doorIdx = self._rand_int(1, height - 2)
            
            if sublevel in [1,3]:
                self.put_obj(Door(color=colors[0], state=1), splitIdx, doorIdx)
            if sublevel == 2:
                self.put_obj(Door(color=colors[0], state=2), splitIdx, doorIdx)
            if sublevel in [4,5,6]:
                self.put_obj(Door(color=colors[0], state=3), splitIdx, doorIdx)
            if sublevel in [3,4]:
                self.put_obj(Key(color=colors[0]), splitIdx-1, doorIdx)
            if sublevel in [5,6]:
                self.place_obj(Key(color=colors[0]), top=(1, 1), size=(splitIdx, height-1))
            if sublevel == 6:
                self.place_obj(Key(color=colors[1]), top=(1, 1), size=(splitIdx, height-1))

            self.put_obj(Goal(color="green", reward=100), width - 2, height//2)

            self.agent_spawn_kwargs = {'size': (splitIdx, height)}


        elif level == 3: #4-way doorkey variants
            goals = random.sample([0,0,0,1], 4)

            goal_positions = [(width-2,height//2), (width//2,1), (1,height//2), (width//2, height-2)]
            door_positions = [(width-2 - 1,height//2), (width//2,0 + 2), (0 + 2,height//2), (width//2, height-2 - 1)]
            subgoal_positions = [(width-2 - 2,height//2), (width//2,0 + 3), (0 + 3,height//2), (width//2, height-2 - 2)]


            # Generate the surrounding walls
            
            if random.randrange(50) > 0:
                self.grid.wall_rect(2, 2, width-4, height-4)
            
            if random.randrange(50) > 45:
                self.grid.wall_rect(1, 1, width-2, height-2)

            # Place doors, goals, subgoals
            for color, goal, goal_p, door, sub in zip(colors, goals, goal_positions, door_positions, subgoal_positions):
                
                if sublevel == 1:
                    self.put_obj(Door(color=color, state=2), door[0], door[1])
                elif sublevel == 2:
                    self.put_obj(Door(color=color, state=3), door[0], door[1])
                    self.place_obj(obj=Key(color=color), top=(3, 3), size=(width-6, height-6))

                self.put_obj(Goal(color='green', reward=50+50*goal), goal_p[0], goal_p[1])

            self.agent_spawn_kwargs = {'top': (2,2), 'size': (width-4, height-4)}


        self.place_agents(**self.agent_spawn_kwargs)
