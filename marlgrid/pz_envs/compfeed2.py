from ..base_AEC import *
from ..objects import *
from random import randrange
import random
import math
from ..puppets import astar
from operator import sub

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

    def pathfind(self, board, start, goal, startDir):
        #returns dict describing directions at tiles along path
        path = astar(board, start, goal)
        pathDict = {}
        for i, pos in enumerate(path):
            if i < len(path)-1:
                pathDict[str(pos)] = [(1, 0), (0, -1), (-1, 0), (0, 1)].index(tuple(map(sub,path[i+1],pos)))
        return pathDict

    def hard_reset(self, **subsets):
        self.boxes = [2, 3, 4, 5]
        self.puppets = [1, 2]
        self.followDistance = [-1,0,1]
        self.adversarial = [True, False]
        self.baits = [1, 2, 3]
        self.informed = ['informed', 'uninformed', 'fake']
        self.hidden = [True, False]
        self.swapType = ['swap', 'replace']
        self.visibility = ['full', 'curtains', 'keys', 'potion']
        self.cause = ['blocks', 'direction', 'accident', 'inability'] #in inability, paths may be blocked for the leader
        self.rational = [True, False]
        self.lava = ['lava', 'block']
        self.lavaHeight = [1,2,3,4]
        self.sharedRewards = [True, False]
        
        # todo: make this clean
        self.boxes = random.choice(self.boxes)
        self.puppets = random.choice(self.puppets)
        self.followDistance = random.choice(self.followDistance)
        self.adversarial = random.choice(self.adversarial)
        self.baits = random.choice(self.baits)
        self.informed = random.choice(self.informed)
        self.visibility = random.choice(self.visibility)
        self.swapType = random.choice(self.swapType)
        self.cause = random.choice(self.cause)
        self.rational = random.choice(self.rational)
        self.lava = random.choice(self.lava)
        self.lavaHeight = random.choice(self.lavaHeight)
        self.sharedRewards = random.choice(self.sharedRewards)
        
        self.startRoom = 2

    def _gen_grid(self, width, height):


        #print(self.boxes)
        self.food_locs = list(range(self.boxes))
        random.shuffle(self.food_locs)
        print(self.lavaHeight)

        #startRoom = 3
        self.release1 = []
        self.release2 = []
        releaseGap = self.boxes*2+1

        self.width = self.boxes*2+3
        self.height = self.lavaHeight+self.startRoom*4+2
        #print(boxes, lavaHeight, width, height)
        self.grid = MultiGrid((self.width, self.height))
        self.grid.wall_rect(0, 0, self.width, self.height)
        #self.grid.wall_rect(0, self.startRoom, width, height-self.startRoom)
        
        print(self.startRoom)
        for j in range(self.width):
            self.put_obj(Wall(), j, self.startRoom*2)
            self.put_obj(Wall(), j, self.startRoom)
            self.put_obj(Wall(), j, self.height-self.startRoom*2-1)
            self.put_obj(Wall(), j, self.height-self.startRoom-1)

        for box in range(self.boxes+1):
            if box < self.boxes:
                self.put_obj(Block(init_state=1, color="blue"), box*2+2, self.startRoom)
                self.release1 += [(box*2+2, self.startRoom)]
                self.put_obj(Block(init_state=1, color="red"), box*2+2, self.startRoom*2)
                self.release2 += [(box*2+2, self.startRoom*2)]
                self.put_obj(Wall(), box*2+1, self.startRoom-1)

                self.put_obj(Block(init_state=1, color="blue"), box*2+2, self.height-self.startRoom-1)
                self.release1 += [(box*2+2, self.height-self.startRoom-1)]
                self.put_obj(Block(init_state=1, color="red"), box*2+2, self.height-self.startRoom*2-1)
                self.release2 += [(box*2+2, self.height-self.startRoom*2-1)]
                self.put_obj(Wall(), box*2+1, self.height-2)
            for j in range(self.lavaHeight):
                x = box*2+1
                y = j+self.startRoom*2+1
                self.put_obj(Lava(), x, y)

        self.init_timers(releaseGap)

        self.agent_spawn_kwargs = {'top': (0,0), 'size': (2, self.width)}
        self.agent_spawn_pos = {'player_0': (1,1,0), 'player_1': (1, self.height-2, 2)}


    def init_timers(self, releaseGap):
        self.timers = {}
        curTime = 1
        for bait in range(self.baits):
            while True:
                baitLength = 7
                if self.informed == "informed":
                    #no hiding
                    swapTime = 1+random.randint(0, baitLength-1)
                elif self.informed == "uninformed":
                    #swap during blind
                    swapTime = random.randint(1, baitLength-2)
                    blindStart = random.randint(0, swapTime)
                    blindStop = random.randint(swapTime, baitLength)
                    self.add_timer("blind", curTime+blindStart)
                    self.add_timer("reveal", curTime+blindStop)
                elif self.informed == "fake":
                    #swap/hide before or after blind
                    if random.choice([True, False]):
                        swapTime = random.randint(1, baitLength)
                        blindStart = random.randint(0, swapTime-2)
                        blindStop = random.randint(blindStart, swapTime-1)
                    else:
                        swapTime = random.randint(0, baitLength-3)
                        blindStart = swapTime+random.randint(swapTime, baitLength-1)
                        blindStop = swapTime+random.randint(blindStart, baitLength)

                    #assert blindStart < blindStop
                    #assert blindStop < baitLength

                    self.add_timer("blind", curTime+blindStart)
                    self.add_timer("reveal", curTime+blindStop)
                if bait == 0:
                    self.add_timer("place", curTime+swapTime)
                else:
                    self.add_timer(self.swapType, curTime+swapTime)

                if bait == self.baits-1 and self.hidden:
                    self.add_timer("hide", curTime+swapTime+1)
                curTime += baitLength
                break
        self.add_timer("release1", curTime+1)
        self.add_timer("release2", curTime+1+releaseGap) #release2 also checks for the x coord of actor/correctness/ends in test mode

    def timer_active(self, name):
        if name == "release1":
            print(self.release1)
            for x,y in self.release1:
                self.del_obj(x,y)
        if name == "release2":
            print(self.release2)
            for x,y in self.release2:
                self.del_obj(x,y)
        if name == "place" or name == "hide":
            if name == "hide":
                pathBox = random.randint(0,self.boxes)
                # todo: if informed, pick favorite known box
            for box in range(self.boxes):
                x = box*2+2
                y = self.height//2#-self.followDistance
                if name == "place":
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
                    if box == pathBox:
                        direction = 0 #player 1 direction
                        pos = (1,1) #player 1 pos
                        print("pathfinding")
                        self.infos["player_1"]["path"] = self.pathfind(self.grid.overlapping, pos, (x,y), direction)
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

        if "blind" in name:
            # record whether each agent can see each food
            pass
        if name in ["place", "swap", "replace"]:
            # record knowledge of where food is (if visible to each agent)
            pass
    	            
