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
        self.followDistance = [-1,0,1]
        self.adversarial = [True, False]
        self.baits = [1, 2, 3]
        self.informed = ['informed', 'uninformed', 'fake']
        self.visibility = ['full', 'curtains', 'keys', 'potion']
        self.cause = ['blocks', 'direction', 'accident', 'inability'] #in inability, paths may be blocked for the leader
        self.rational = [True, False]
        self.lava = ['lava', 'block']
        self.lavaHeight = [1,2,3]
        
        # todo: make this clean
        self.boxes = random.choice(self.boxes)
        self.puppets = random.choice(self.puppets)
        self.followDistance = random.choice(self.followDistance)
        self.adversarial = random.choice(self.adversarial)
        self.baits = random.choice(self.baits)
        self.informed = random.choice(self.informed)
        self.visibility = random.choice(self.visibility)
        self.cause = random.choice(self.cause)
        self.rational = random.choice(self.rational)
        self.lava = random.choice(self.lava)
        self.lavaHeight = random.choice(self.lavaHeight)
        
        self.startRoom = 3

    def _gen_grid(self, width, height):

        self.timers = {}

        #print(self.boxes)
        self.food_locs = list(range(self.boxes))
        random.shuffle(self.food_locs)
        print(self.food_locs)

        #startRoom = 3

        width = self.boxes*2+3
        height = self.lavaHeight+self.startRoom*2+1
        #print(boxes, lavaHeight, width, height)
        self.grid = MultiGrid((width, height))
        self.grid.wall_rect(0, 0, width, height)
        
        for j in range(width):
            self.put_obj(Wall(), j, self.startRoom-1)
            self.put_obj(Wall(), j, height-self.startRoom)

        for box in range(self.boxes+1):
            if box < self.boxes:
                self.put_obj(Block(init_state=1), box*2+2, self.startRoom-1)
                self.put_obj(Block(init_state=1), box*2+2, height-self.startRoom)
            for j in range(self.lavaHeight+1):
                x = box*2+1
                y = j+self.startRoom
                self.put_obj(Lava(), x, y)

        self.agent_spawn_kwargs = {'top': (0,0), 'size': (2, width)}
        self.agent_spawn_pos = {'player_0': (1,1,0), 'player_1': (1, height-2, 2)}
        
        self.timers["bait"] = 2
        self.timers["replace"] = 3
        self.timers["hide"] = 4
        self.timers["release2"] = 5

        #todo:
        '''
        blind - both sides
            control with informed param
        initial cage - this controls blindness and has invis potion
        random-swap - just reshuffles all foods
        release1
        
        puppet controller - goto specified tile with a*?
        puppet controller - keep track of where it thinks foods are

        agent - test which food we went do and which in food_locs it was
        '''
        #print([x.dir for x in self.agent_instances])

    def timer_active(self, name):
        if name == "release2":
            for box in range(self.boxes):
                self.del_obj(box*2+2, self.startRoom-1)
                self.del_obj(box*2+2, self.height-self.startRoom)
        if name == "bait" or name == "hide":
    	    for box in range(self.boxes):
                x = box*2+2
                y = self.height//2#-self.followDistance
                if name == "bait":
                    if box == self.food_locs[0]:
                        reward = 100
                        size = 1
                        self.put_obj(Goal(reward=reward, size=size, color='green'), x, y)
                    elif box == self.food_locs[1]:
                        reward = 25
                        size = 0.5
                        self.put_obj(Goal(reward=reward, size=size, color='green'), x, y)
    	            
                elif name == "hide":
                    b1 = Box(color="yellow")
                    c = self.grid.get(x,y)
                    if c:
        	            b1.contains = c
        	            b1.can_overlap = c.can_overlap
        	            b1.get_reward = c.get_reward
                    else:
                        b1.can_overlap = lambda : True
                        b1.get_reward = lambda x: 0
                        #todo: why does one of these have arg? overlap is property?
                    self.put_obj(b1, x, y)
        if name == "replace":          
            #swap big food with a no food tile 
            for box in range(self.boxes):
                x = box*2+2
                y = self.height//2#-self.followDistance
                if box == self.food_locs[2]:
                    reward = 100
                    size = 1
                    self.put_obj(Goal(reward=reward, size=size, color='green'), x, y)
                elif box == self.food_locs[0]:
                    self.del_obj(x,y)
        if name == "swap":
            for box in range(self.boxes):
                x = box*2+2
                y = self.height//2#-self.followDistance
                if box == self.food_locs[1]:
                    reward = 100
                    size = 1
                    self.put_obj(Goal(reward=reward, size=size, color='green'), x, y)
                elif box == self.food_locs[0]:
                    reward = 25
                    size = 0.5
                    self.put_obj(Goal(reward=reward, size=size, color='green'), x, y)
    	            
