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
    random_mode = True
    stages = ["1a", "1b", "1c", "1d", "2a", "2b", "2c", "2d", "2e", "2f", "3a", "3b"]
    curStage = 0;
    loading_mode = False
    saving_mode = False
    path = ''

    def _rand_int(self, x, y):
        return randrange(x, y)

    def _gen_grid(self, width, height):
        self.grid = MultiGrid((width, height))
        if self.random_mode:
            stage = random.choice(self.stages)
        else:
            stage = self.stages[self.curStage]
            self.curStage = (self.curStage + 1) % len(self.stages) 

        colors = random.sample(['purple','orange','yellow','blue','pink','red'], 4)

        #grid and surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        if "1" in stage: #empty, cluttered, and mazes
            #self.put_obj(Goal(color="green", reward=100), width - 2, height//2)

            if stage[1] in "bcd":
                for x in range(2, width-1, 2):
                    for y in range(2, height-1, 2):
                        if stage[1] in "cd":
                            self.put_obj(Wall(), x, y)
                        else:
                            num = randrange(0,10)
                            if num == 0:
                                self.put_obj(Wall(), x, y)
                            elif num == 1:
                                self.put_obj(Door(color=colors[0], state=randrange(1,4)), x, y)
                            elif num == 2:
                                self.put_obj(Key(color=colors[0], state=randrange(1,4)), x, y)
            if stage[1] == "d":
                for i in range(3):
                    num = randrange(0,3)
                    if num == 0:
                        self.place_obj(Wall(), top=(0, 0), size=(width, height))
                    elif num == 1:
                        self.place_obj(Door(color=colors[0], state=randrange(1,4)), top=(0, 0), size=(width, height))
                    elif num == 2:
                        self.place_obj(Key(color=colors[0], state=randrange(1,4)), top=(0, 0), size=(width, height))

            self.place_obj(Goal(color="green", reward=100), top=(0, 0), size=(width, height))


        elif "2" in stage: #doorkey variants
            splitIdx = self._rand_int(2, width - 2)
            self.grid.vert_wall(splitIdx, 1)

            doorIdx = self._rand_int(1, height - 2)
            
            if stage[1] in "ac":
                self.put_obj(Door(color=colors[0], state=1), splitIdx, doorIdx)
            if stage[1] == "b":
                self.put_obj(Door(color=colors[0], state=2), splitIdx, doorIdx)
            if stage[1] in "def":
                self.put_obj(Door(color=colors[0], state=3), splitIdx, doorIdx)
            if stage[1] in "cd":
                self.put_obj(Key(color=colors[0]), splitIdx-1, doorIdx)
            if stage[1] in "ef":
                self.place_obj(Key(color=colors[0]), top=(1, 1), size=(splitIdx, height-1))
            if stage[1] == "f":
                self.place_obj(Key(color=colors[1]), top=(1, 1), size=(splitIdx, height-1))

            self.put_obj(Goal(color="green", reward=100), width - 2, height//2)

            self.agent_spawn_kwargs = {'size': (splitIdx, height)}


        elif "3" in stage: #4-way doorkey variants
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
                
                if stage[1] == "a":
                    self.put_obj(Door(color=color, state=2), door[0], door[1])
                elif stage[1] == "b":
                    self.put_obj(Door(color=color, state=3), door[0], door[1])
                    self.place_obj(obj=Key(color=color), top=(3, 3), size=(width-6, height-6))

                self.put_obj(Goal(color='green', reward=50+50*goal), goal_p[0], goal_p[1])

            self.agent_spawn_kwargs = {'top': (2,2), 'size': (width-4, height-4)}


        self.place_agents(**self.agent_spawn_kwargs)
