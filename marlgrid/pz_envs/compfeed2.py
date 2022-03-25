from ..base_AEC import *
from ..objects import *
from random import randrange
import random
import math
from ..puppets import astar, pathfind

class para_Mindreading(para_MultiGridEnv):

    mission = "get to the goal"
    metadata = {'render.modes': ['human', 'rgb_array'], "name": "mindreadingEnv"}
    variants = []

    def hard_reset(self, params=None):
        
        defaults = {
                "adversarial": [True, False],
                "hidden": [True, False],
                "rational": [True, False],
                "sharedRewards": [True, False],
                "boxes": [2,3,4,5],
                "puppets": [1,2],
                "followDistance": [-1,0,1],
                "lavaHeight": [1,2,3,4],
                "baits": [1,2],
                "informed": ['informed', 'uninformed', 'fake'],
                "swapType": ['swap', 'replace'],
                "visibility": ['full', 'curtains', 'keys', 'potion'],
                "cause": ['blocks', 'direction', 'accident', 'inability'],
                "lava": ['lava', 'block'],
                }

        if params == None:
            params = {}
        for k in defaults.keys():
            if k in params.keys():
                params[k] = random.choice(params[k])
            else:
                params[k] = random.choice(defaults[k])
        self.params = params

    def reset_vision(self):
        boxes = self.params["boxes"]
        for agent in self.agents:
            self.agent_goal[agent] = random.choice(range(boxes))
            self.best_reward[agent] = -100
            for box in range(boxes):
                self.last_seen_reward[agent+str(box)] = -100
                if agent+str(box) not in self.can_see.keys():
                    self.can_see[agent+str(box)] = True #default to not hidden until it is

    def _gen_grid(self, width, height,
                adversarial=True,
                hidden=True,
                rational=True,
                sharedRewards=False,
                boxes=3,
                puppets=1,
                followDistance=0,
                lavaHeight=3,
                baits=1,
                informed='informed',
                swapType='swap',
                visibility='full',
                cause='blocks',
                lava='lava',
                ):
        startRoom = 2
        atrium = 3
        self.food_locs = list(range(boxes))
        random.shuffle(self.food_locs)
        self.release1 = []
        self.release2 = []
        releaseGap = boxes*2+atrium
        self.width = boxes*2+3
        self.height = lavaHeight+startRoom*2+atrium*2+2
        self.grid = MultiGrid((self.width, self.height))
        self.grid.wall_rect(0, 0, self.width, self.height)

        for j in range(self.width):
            self.put_obj(Wall(), j, startRoom+atrium)
            self.put_obj(Wall(), j, startRoom)
            self.put_obj(Wall(), j, self.height-startRoom-atrium-1)
            self.put_obj(Wall(), j, self.height-startRoom-1)

        for box in range(boxes+1):
            if box < boxes:
                self.put_obj(Block(init_state=1, color="blue"), box*2+2, startRoom)
                self.release1 += [(box*2+2, startRoom)]
                self.put_obj(Block(init_state=1, color="red"), box*2+2, startRoom+atrium)
                self.release2 += [(box*2+2, startRoom+atrium)]
                self.put_obj(Wall(), box*2+1, startRoom-1)

                self.put_obj(Block(init_state=1, color="blue"), box*2+2, self.height-startRoom-1)
                self.release1 += [(box*2+2, self.height-startRoom-1)]
                self.put_obj(Block(init_state=1, color="red"), box*2+2, self.height-startRoom-atrium-1)
                self.release2 += [(box*2+2, self.height-startRoom-atrium-1)]
                self.put_obj(Wall(), box*2+1, self.height-2)
            for j in range(lavaHeight):
                x = box*2+1
                y = j+startRoom+atrium+1
                self.put_obj(Lava(), x, y)

        self.agent_spawn_kwargs = {'top': (0,0), 'size': (2, self.width)}
        self.agent_spawn_pos = {'player_0': (1,1,0), 'player_1': (1, self.height-2, 2)}

        self.agent_goal, self.last_seen_reward, self.can_see, self.best_reward = {}, {}, {}, {}
        self.reset_vision()
        # init timers
        

        self.timers = {}
        curTime = 1
        self.add_timer("init", 1)
        for bait in range(baits):
            while True:
                baitLength = 7
                if informed == "informed":
                    #no hiding
                    swapTime = 1+random.randint(0, baitLength-1)
                elif informed == "uninformed":
                    #swap during blind
                    swapTime = random.randint(1, baitLength-2)
                    blindStart = random.randint(0, swapTime)
                    blindStop = random.randint(swapTime, baitLength)
                    self.add_timer("blind player_1", curTime+blindStart)
                    self.add_timer("reveal player_1", curTime+blindStop)
                elif informed == "fake":
                    #swap/hide before or after blind
                    if random.choice([True, False]):
                        swapTime = random.randint(1, baitLength)
                        blindStart = random.randint(0, swapTime-2)
                        blindStop = random.randint(blindStart, swapTime-1)
                    else:
                        swapTime = random.randint(0, baitLength-3)
                        blindStart = swapTime+random.randint(swapTime, baitLength-1)
                        blindStop = swapTime+random.randint(blindStart, baitLength)

                    assert blindStart < blindStop
                    assert blindStop < baitLength

                    self.add_timer("blind player_1", curTime+blindStart)
                    self.add_timer("reveal player_1", curTime+blindStop)
                if bait == 0:
                    self.add_timer("place", curTime+swapTime)
                else:
                    self.add_timer(swapType, curTime+swapTime)

                if bait == baits-1:
                    if hidden:
                        self.add_timer("hide", curTime+swapTime+1)
                curTime += baitLength
                break
        self.add_timer("release1", curTime+1)
        self.add_timer("release2", curTime+1+releaseGap) #release2 also checks for the x coord of actor/correctness/ends in test mode

    def timer_active(self, name):
        boxes = self.params["boxes"]
        y = self.height//2#-self.followDistance
        if name == "release1":
            for x,y in self.release1:
                self.del_obj(x,y)
        if name == "release2":
            for x,y in self.release2:
                self.del_obj(x,y)
        if name == "place" or name == "hide":
            for box in range(boxes):
                x = box*2+2
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
        if name == "replace":          
            #swap big food with a no food tile 
            for box in range(boxes):
                x = box*2+2
                y = self.height//2#-self.followDistance
                if box == self.food_locs[2]:
                    reward = 100
                    size = 1
                    self.put_obj(Goal(reward=reward, size=size, color='green'), x, y)
                elif box == self.food_locs[0]:
                    self.del_obj(x,y)
        if name == "swap":
            for box in range(boxes):
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

        if "blind" in name or "reveal" in name:
            # record whether each agent can see each food
            print(name)
            splitName = name.split()
            agent = self.instance_from_name[splitName[1]]
            for box in range(boxes):
                self.can_see[splitName[1] + str(box)] = False if "blind" in name else True

        # whenever food updates, remember locations
        if name in ["init", "place", "swap", "replace", "reveal"]:
            print(name)

            self.reset_vision()

            for box in range(boxes):
                x = box*2+2
                for agent in self.agents:
                    if self.can_see[agent+str(box)]:
                        tile = self.grid.get(x,y)
                        if hasattr(tile, "reward") and hasattr(tile, "size"):
                            #size used to distinguish treats from boxes
                            self.last_seen_reward[agent+str(box)] = tile.reward
                            #print('rew update', agent, box, tile.reward)
            new_target = False
            for box in range(boxes):
                for agent in self.agents:
                    reward = self.last_seen_reward[agent+str(box)]
                    if (self.agent_goal[agent] != box) and (reward >= self.best_reward[agent]):
                        self.agent_goal[agent] = box
                        self.best_reward[agent] = reward
                        #print('found box', name, agent, box, reward)
                        new_target = True
                        target_agent = agent
            if new_target and target_agent != "player_0":
                a = self.instance_from_name[target_agent]
                x = self.agent_goal[agent]*2+2
                print("pathfinding to", self.agent_goal[agent], x, y)
                path = pathfind(self.grid.overlapping, a.pos, (x, y), a.dir)
                self.infos[agent]["path"] = path
                print(path)
            

    	            
