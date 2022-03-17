from ..base_AEC import *
from ..objects import *
from random import randrange
import random
import math

class para_Mindreading(para_MultiGridEnv):

    mission = "get to the goal"
    metadata = {'render.modes': ['human', 'rgb_array'], "name": "mindreadingEnv"}
    variants = []

    boxes = [2, 3, 4, 5]
    puppets = [1, 2]
    leader = ['follow', 'lead', 'longfollow']
    adversarial = [True, False]
    baits = [1, 2, 3]
    informed = ['informed', 'uninformed', 'fake']
    visibility = ['full', 'curtains', 'keys', 'potion']
    cause = ['blocks', 'direction', 'accident', 'inability'] #in inability, paths may be blocked for the leader
    rational = [True, False]
    lava = ['lava', 'block']
    lavaHeight = [1,3,5]

    def hard_reset(self, **kwargs):
        self.boxes = [2, 3, 4, 5]
        self.puppets = [1, 2]
        self.leader = ['follow', 'lead', 'longfollow']
        self.adversarial = [True, False]
        self.baits = [1, 2, 3]
        self.informed = ['informed', 'uninformed', 'fake']
        self.visibility = ['full', 'curtains', 'keys', 'potion']
        self.cause = ['blocks', 'direction', 'accident', 'inability'] #in inability, paths may be blocked for the leader
        self.rational = [True, False]
        self.lava = ['lava', 'block']
        self.lavaHeight = [1,2,3]

    def _gen_grid(self, width, height):

        self.timers = {}

        # todo: make this clean
        boxes = random.choice(self.boxes)
        puppets = random.choice(self.puppets)
        leader = random.choice(self.leader)
        adversarial = random.choice(self.adversarial)
        baits = random.choice(self.baits)
        informed = random.choice(self.informed)
        visibility = random.choice(self.visibility)
        cause = random.choice(self.cause)
        rational = random.choice(self.rational)
        lava = random.choice(self.lava)
        lavaHeight = random.choice(self.lavaHeight)

        food_loc = random.choice(range(boxes))

        startRoom = 3

        width = boxes*2+3
        height = lavaHeight+startRoom*2+1
        print(boxes, lavaHeight, width, height)
        self.grid = MultiGrid((width, height))
        self.grid.wall_rect(0, 0, width, height)
        
        for j in range(width):
            self.put_obj(Wall(), j, startRoom-1)
            self.put_obj(Wall(), j, height-startRoom)

        for box in range(boxes+1):
            self.put_obj(Block(init_state = 1), box*2+1, startRoom-1)
            self.put_obj(Block(init_state = 1), box*2+1, height-startRoom)
            for j in range(lavaHeight+1):
                x = box*2+1
                y = j+startRoom
                self.put_obj(Lava(), x, y)

        self.agent_spawn_kwargs = {'top': (0,0), 'size': (2, width)}
        self.agent_spawn_pos = {'player_0': (1,1,0), 'player_1': (1, height-2, 2)}
        #print([x.dir for x in self.agent_instances])

    def timer_active(self, name):
        pass