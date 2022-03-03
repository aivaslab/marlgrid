from ..base_AEC import *
from ..objects import *
from random import randrange
import random
import math

class para_CompFeedEnv(para_MultiGridEnv):

    mission = "get to the goal"
    metadata = {'render.modes': ['human', 'rgb_array'], "name": "compfeedenv"}
    variants = []

    def _gen_grid(self, width, height):
        width = 15
        height = 9
        #curType = "informed" #both glass, food baited, both released
        #curType = "uninformed" #both glass, dom blinded, food baited, both released
        #curType = "control misinformed" #both glass, food baited, food swapped, both released
        #curType = "misinformed" #both glass, food baited, dom blinded, food swapped, both released
        curType = random.choice(["informed","uninformed","control misinformed","misinformed"])

        self.timers = {}

        self.timers["bait"] = 3
        self.timers["hide"] = 4
        self.timers["release"] = 5
        if curType == "uninformed":
            self.timers["dBlind"] = 2
        if curType == "misinformed":
            self.timers["dBlind"] = 6
            self.timers["swap"] = 7
            self.timers["hide"] = 8
            self.timers["release"] = 9
        if curType == "control misinformed":
            #self.timers["dBlind"] = (6)
            self.timers["swap"] = 7
            self.timers["hide"] = 8
            self.timers["release"] = 9

        self.food_loc = random.choice([0,1])


        self.grid = MultiGrid((width, height))
        self.grid.wall_rect(0, 0, width, height)
        self.grid.wall_rect(0, 3, 3, 3)

        self.grid.vert_wall(2, 1)
        self.put_obj(GlassBlock(color='blue', init_state=1), 2, 4)
        self.put_obj(Lava(), 4, 4)

        #test block

        # left action = 0, right = 1
        self.put_obj(Tester(color='red', correct_direction = 0+self.food_loc), 3, 4)

        self.grid.wall_rect(9, 3, 3, 3)
        self.grid.vert_wall(9, 1)
        self.put_obj(GlassBlock(color='blue', init_state=1), 9, 4) 


        #puppet guiding arrow
        
        if curType == "informed" or curType == "misinformed":
            direction = 1+self.food_loc*2
        elif curType == "control misinformed":
            direction = 3-(self.food_loc)*2
        else:
            direction = random.choice([1,3])
        self.put_obj(Arrow(direction=direction), 8, 4)

        for x in range(7,9):
            for y in range(1,4):
                self.put_obj(Lava(), x, y)
                self.put_obj(Lava(), x, y+4)
                
        self.put_obj(Lava(), 7, 4)
        self.del_obj(8,3)
        self.del_obj(8,2)
        self.del_obj(7,2)
        self.del_obj(8,5)
        self.del_obj(8,6)
        self.del_obj(7,6)
        self.put_obj(Arrow(direction=0), 8, 2)
        self.put_obj(Arrow(direction=0), 8, 6)

        self.agent_spawn_kwargs = {'top': (0,3), 'size': (2, 2)}
        self.agent_spawn_pos = {'player_0': (1,4,0), 'player_1': (10, 4, 2)}
        #todo: place puppet agent too
        '''for agent in self.agent_instances:
        	if agent.name == 'player_1':
        		agent.dir = 2
        	if agent'''
        print([x.dir for x in self.agent_instances])

        #self.place_agents(**self.agent_spawn_kwargs)

    def timer_active(self, name):
        if name == "bait":
            if self.food_loc == 0: #6, 2 and 6, 6
                self.put_obj(Goal(color='green', reward=100), 6, 2)
                self.put_obj(Goal(color='green', reward=50, size=0.5), 6, 6)
            else:
                self.put_obj(Goal(color='green', reward=100), 6, 6)
                self.put_obj(Goal(color='green', reward=50, size=0.5), 6, 2)
        elif name == "hide":
            b1 = Box(color='yellow')
            b1.contains = self.grid.get(6,2)
            b1.can_overlap = b1.contains.can_overlap
            b1.get_reward = b1.contains.get_reward
            b2 = Box(color='yellow')
            b2.contains = self.grid.get(6,6)
            b2.can_overlap = b2.contains.can_overlap
            b2.get_reward = b2.contains.get_reward
            self.put_obj(b1, 6, 2)
            self.put_obj(b2, 6, 6)
        elif name == "release":
            self.del_obj(2, 4) #should change to transparant mode?
            self.del_obj(9, 4)
        elif name == "dBlind":
            self.put_obj(Block(color='red', init_state=1), 9, 4) 
        if name == "swap":
            # swap foods and arrow
            if self.food_loc == 0: #6, 2 and 6, 6
                self.put_obj(Goal(color='green', reward=100), 6, 6)
                self.put_obj(Goal(color='green', reward=50, size=0.5), 6, 2)
            else:
                self.put_obj(Goal(color='green', reward=100), 6, 2)
                self.put_obj(Goal(color='green', reward=50, size=0.5), 6, 6)
            
            
            #todo: add intro to food hiding in boxes
            #right after food of different sizes
            
            #add control scenario where block goes solid and then no swap happens
