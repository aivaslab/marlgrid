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
    metadata = {'render_modes': ['human', 'rgb_array'], "name": "tutorial"}
    random_mode = True
    seed_mode = False
    curSeed = 1337
    variants = ["1a", "1b", "1c", "1d", "2a", "2b", "2c", "2d", "2e", "2f", "3a", "3b"]
    curvariant = 0;
    loading_mode = False
    saving_mode = False
    path = ''
    #todo: add a 'food is hidden in x box' tutorial
    # how to demonstrate? animation? lever that flips all Red/Blue boxes like in mario?
    #plus automated puppet to trigger the lever on a timer ! well that's just all timelines. 
    #make a timeline puppet that just changes state

    def _init_level1(self, variant, width, height):
        if variant[1] in "bcd":
            for x in range(2, width-1, 2):
                for y in range(2, height-1, 2):
                    if variant[1] in "cd":
                        self.put_obj(Wall(), x, y)
                    else:
                        num = randrange(0,10)
                        if num == 0:
                            self.put_obj(Wall(), x, y)
                        elif num == 1:
                            self.put_obj(Door(color=colors[0], state=randrange(1,4)), x, y)
                        elif num == 2:
                            self.put_obj(Key(color=colors[0], state=randrange(1,4)), x, y)
        if variant[1] == "d":
            for i in range(3):
                num = randrange(0,3)
                if num == 0:
                    self.place_obj(Wall(), top=(0, 0), size=(width, height))
                elif num == 1:
                    self.place_obj(Door(color=colors[0], state=randrange(1,4)), top=(0, 0), size=(width, height))
                elif num == 2:
                    self.place_obj(Key(color=colors[0], state=randrange(1,4)), top=(0, 0), size=(width, height))

        self.place_obj(Goal(color="green", reward=100), top=(0, 0), size=(width, height))
        if variant[1] == "e":
            self.place_obj(Goal(color="green", reward=50, size=0.5), top=(0, 0), size=(width, height))

    def _set_seed(self, seed):
        if seed != -1:
            self.seed_mode = True
            self.curSeed = seed

    def _rand_int(self, x, y):
        return randrange(x, y)

    def _gen_grid(self, width, height):
        self.grid = MultiGrid((width, height))
        if self.seed_mode:
            random.seed(self.curSeed)
            self.curSeed += 1

        if self.random_mode:
            variant = random.choice(self.variants)
        else:
            variant = self.variants[self.curvariant]
            self.curvariant = (self.curvariant + 1) % len(self.variants) 

        colors = random.sample(['purple','orange','yellow','blue','pink','red'], 4)

        #grid and surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        if "1" in variant: #empty, cluttered, and mazes
            #self.put_obj(Goal(color="green", reward=100), width - 2, height//2)
            self._init_level1(variant, width=width, height=height)

        elif "2" in variant: #doorkey variants
            splitIdx = self._rand_int(2, width - 2)
            self.grid.vert_wall(splitIdx, 1)

            doorIdx = self._rand_int(1, height - 2)
            
            if variant[1] in "ac":
                self.put_obj(Door(color=colors[0], state=1), splitIdx, doorIdx)
            if variant[1] == "b":
                self.put_obj(Door(color=colors[0], state=2), splitIdx, doorIdx)
            if variant[1] in "def":
                self.put_obj(Door(color=colors[0], state=3), splitIdx, doorIdx)
            if variant[1] in "cd":
                self.put_obj(Key(color=colors[0]), splitIdx-1, doorIdx)
            if variant[1] in "ef":
                self.place_obj(Key(color=colors[0]), top=(1, 1), size=(splitIdx, height-1))
            if variant[1] == "f":
                self.place_obj(Key(color=colors[1]), top=(1, 1), size=(splitIdx, height-1))

            self.put_obj(Goal(color="green", reward=100), width - 2, height//2)

            self.agent_spawn_kwargs = {'size': (splitIdx, height)}


        elif "3" in variant: #4-way doorkey variants
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
                
                if variant[1] == "a":
                    self.put_obj(Door(color=color, state=2), door[0], door[1])
                elif variant[1] == "b":
                    self.put_obj(Door(color=color, state=3), door[0], door[1])
                    self.place_obj(obj=Key(color=color), top=(3, 3), size=(width-6, height-6))

                self.put_obj(Goal(color='green', reward=50+50*goal), goal_p[0], goal_p[1])

            self.agent_spawn_kwargs = {'top': (2,2), 'size': (width-4, height-4)}

        elif "4" in variant: #memory

            self.grid.wall_rect(0, 1, width-1, height-2)

            goals = random.sample([0, 1], 2)


            self.put_obj(Lava(), 4, 4)
            self.put_obj(Lava(), 5, 4)

            for x in range(2,6):
                self.put_obj(Lava(), x, 5)
                self.put_obj(Lava(), x, 3)

            if variant[1] in 'a': #no visible goal
                egg = True
            if variant[1] in 'bcd':
                self.put_obj(Goal(reward=100, color='green'), 6, 4)
            if variant[1] in 'efgh': #offset goal
                self.put_obj(Goal(reward=100, color='green'), 6, 3+2*goals[0])
            if variant[1] in 'cd': #one path blocked by lava
                self.put_obj(Lava(), 6, 3+2*goals[0])
                self.put_obj(Lava(), 6, 2+4*goals[0])
            if variant[1] in 'd': # alt path blocked by lava
                self.put_obj(Lava(), 6, 3+2*goals[1])
                self.put_obj(Lava(), 6, 2+4*goals[1])
            if variant[1] in 'g': # alt offset goal (smaller)
                self.put_obj(Goal(reward=50, color='green', size=0.5), 6, 3+2*goals[1])
            if variant[1] in 'fh': #lava btw offsets
                self.put_obj(Lava(), 6, 5)

            self.agent_spawn_kwargs = {'top': (3,4), 'size': (1, 1)}

        #self.place_agents(**self.agent_spawn_kwargs)


