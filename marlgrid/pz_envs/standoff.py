from ..base_AEC import *
from ..objects import *
from random import randrange
import random
import math
from ..puppets import astar, pathfind
import copy
from marlgrid.marlgrid.pz_envs.scenario_configs import ScenarioConfigs


class para_standoffEnv(para_MultiGridEnv):

    mission = "get the best food before your opponent"
    metadata = {'render_modes': ['human', 'rgb_array'], "name": "standoffEnv"}
    configs = ScenarioConfigs.standoff

    def hard_reset(self, params=None):
        defaults = {
                "adversarial": [True],
                "hidden": [True],
                "rational": [True],
                "sharedRewards": [False],
                "firstBig": [True],
                "boxes": [5],#[2,3,4,5],
                "puppets": [1],
                "followDistance": [0], #0 = d first, 1=sub first
                "lavaHeight": [2],
                "baits": [1],
                "baitSize": [2],
                "informed": ['informed'],
                "swapType": ['swap'],
                "visibility": ['curtains'], #keys, invisibility potion
                "cause": ['blocks', 'direction', 'accident', 'inability'],
                "lava": ['lava', 'block'],
                }

        newParams = copy.copy(params)
        if params == None:
            params = {}
        for k in defaults.keys():
            if k in params.keys():
                if isinstance(params[k], list):
                    newParams[k] = random.choice(params[k])
            else:
                newParams[k] = random.choice(defaults[k])
        self.params = newParams

    def reset_vision(self):
        boxes = self.params["boxes"]
        for agent in self.agents_and_puppets():
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
                boxes=5,
                puppets=1,
                followDistance=0,
                lavaHeight=2,
                baits=1,
                baitSize=2,
                informed='informed',
                swapType='swap',
                visibility='curtains',
                cause='blocks',
                lava='lava',
                firstBig=True,
                ):
        startRoom = 2
        atrium = 2
        
        if swapType == "replace" and boxes <=2:
            swapType = "swap"
        
        self.box_reward = 1
        self.food_locs = list(range(boxes))
        random.shuffle(self.food_locs)
        self.release1 = []
        self.release2 = []
        releaseGap = boxes*2+atrium
        self.width = boxes*2+3
        self.height = lavaHeight+startRoom*2+atrium*2+2
        self.grid = MultiGrid((self.width, self.height))
        self.grid.wall_rect(1, 1, self.width-2, self.height-2)
        
        self.agent_spawn_kwargs = {'top': (0,0), 'size': (2, self.width)}
        self.agent_spawn_pos = {}
        self.agent_box_pos = {}
        for k, agent in enumerate(self.agents_and_puppets()):
            h = 1 if agent == "player_0" else self.height-2
            d = 1 if agent == "player_0" else 3
            xx = 2*random.choice(range(boxes))+2
            self.agent_spawn_pos[agent] = (xx, h, d)
            self.agent_box_pos[agent] = (xx, h + (1 if agent == "player_0" else -1))
            a = self.instance_from_name[agent]
            if k > puppets:
                a.spawn_delay = 1000
                a.active = False

        for j in range(self.width):
            self.put_obj(Wall(), j, startRoom+atrium)
            self.put_obj(Wall(), j, startRoom)
            self.put_obj(Wall(), j, self.height-startRoom-atrium-1)
            self.put_obj(Wall(), j, self.height-startRoom-1)
        
        for j in range(2,self.width-2):
            if visibility == "curtains":
                for i in range(startRoom+1, startRoom+atrium):
                    self.put_obj(Curtain(color='red'), j, i)
                for i in range(self.height-startRoom-atrium-1+1, self.height-startRoom-1):
                    self.put_obj(Curtain(color='red'), j, i)
                    
        self.grid.wall_rect(0, 0, self.width, self.height)

        for box in range(boxes+1):
            if box < boxes:
                self.put_obj(Block(init_state=0, color="blue"), box*2+2, startRoom)
                self.release1 += [(box*2+2, startRoom)]
                self.put_obj(Block(init_state=0, color="blue"), box*2+2, startRoom+atrium)
                self.release2 += [(box*2+2, startRoom+atrium)]
                self.put_obj(Wall(), box*2+1, startRoom-1)

                self.put_obj(Block(init_state=0, color="blue"), box*2+2, self.height-startRoom-1)
                self.release1 += [(box*2+2, self.height-startRoom-1)]
                self.put_obj(Block(init_state=0, color="blue"), box*2+2, self.height-startRoom-atrium-1)
                self.release2 += [(box*2+2, self.height-startRoom-atrium-1)]
                self.put_obj(Wall(), box*2+1, self.height-2)
            for j in range(lavaHeight):
                x = box*2+1
                y = j+startRoom+atrium+1
                self.put_obj(GlassBlock(color="cyan", init_state=1), x, y)

        self.agent_goal, self.last_seen_reward, self.can_see, self.best_reward = {}, {}, {}, {}
        self.reset_vision()
        # init timers

        self.timers = {}
        curTime = 1
        self.add_timer("init", 1)
        for bait in range(0, baits*baitSize, baitSize):
            baitLength = 7
            informed2 = informed
            if informed == "half1":
                informed2 = "informed" if bait == 0 else "uninformed"
            elif informed == "half2":
                informed2 = "informed" if bait == 1 else "uninformed"
                
            if informed2 == "informed":
                #no hiding
                swapTime = random.randint(1, baitLength-1)
            elif informed2 == "uninformed":
                #swap during blind
                swapTime = random.randint(1, baitLength-2)
                blindStart = random.randint(0, swapTime)
                blindStop = random.randint(swapTime, baitLength)
                self.add_timer("blind player_1", curTime+blindStart)
                self.add_timer("reveal player_1", curTime+blindStop)
            elif informed2 == "fake":
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
            if bait < 2:
                if baitSize == 2:
                    self.add_timer("place12", curTime+swapTime)
                elif baitSize == 1:
                    if firstBig == bait:
                        self.add_timer("place1", curTime+swapTime)
                    else:
                        self.add_timer("place2", curTime+swapTime)
            else:
                st = swapType
                if "remove" in st:
                    st = st + random.choice(["1","2"])
                self.add_timer(st, curTime+swapTime)
            if hidden:                    
                if bait+baitSize < 2:
                    if firstBig == bait:
                        self.add_timer("hide1", curTime+swapTime+1)
                    else:
                        self.add_timer("hide2", curTime+swapTime+1)
                if bait+baitSize > baits-1:
                        self.add_timer("hideall", curTime+swapTime+1)
            curTime += baitLength
        self.add_timer("release1", curTime+1)
        self.add_timer("release2", curTime+1+releaseGap) #release2 also checks for the x coord of actor/correctness/ends in test mode

    def timer_active(self, name):
        boxes = self.params["boxes"]
        firstBig = self.params["firstBig"]
        followDistance = self.params["followDistance"]
        y = self.height//2-followDistance
        if name == "release1":
            for xx,yy in self.release1:
                self.del_obj(xx,yy)
        if name == "release2":
            for xx,yy in self.release2:
                self.del_obj(xx,yy)
        if "place" in name or "hide" in name or "remove" in name:
            for box in range(boxes):
                x = box*2+2
                if "place" in name:
                    if box == self.food_locs[0] and "1" in name:
                        self.put_obj(Goal(reward=100, size=1.0, color='green'), x, y)
                    if box == self.food_locs[1] and "2" in name:
                        self.put_obj(Goal(reward=25, size=0.5, color='green'), x, y)
    	            
                elif "hide" in name:
                    if "all" in name or (box == self.food_locs[0] and "1" in name) or (box == self.food_locs[1] and "2" in name):
                        b1 = Box(color="yellow")
                        c = self.grid.get(x,y)
                        if c:
            	            b1.contains = c
            	            b1.can_overlap = c.can_overlap
            	            b1.get_reward = c.get_reward
                        else:
                            b1.can_overlap = lambda : True
                            b1.get_reward = lambda x: self.box_reward
                            #todo: why does one of these have arg? overlap is property?
                        self.put_obj(b1, x, y)
                        
                elif "remove" in name:
                    if box == self.food_locs[0] and "1" in name:
                        self.del_obj(x,y)
                    elif box == self.food_locs[1] and "2" in name:
                        self.del_obj(x,y)
        if name == "replace":          
            #swap big food with a no food tile
            #currently only does big food, should it do small? 
            for box in range(boxes):
                x = box*2+2
                y = self.height//2-followDistance
                if box == self.food_locs[2]:
                    self.put_obj(Goal(reward=100, size=1.0, color='green'), x, y)
                elif box == self.food_locs[0]:
                    self.del_obj(x,y)
        if name == "move":          
            #both foods are moved to new locations
            for box in range(boxes):
                x = box*2+2
                if box == self.food_locs[2]:
                    self.put_obj(Goal(reward=100, size=1.0, color='green'), x, y)
                if box == self.food_locs[3]:
                    self.put_obj(Goal(reward=25, size=0.5, color='green'), x, y)
                elif box == self.food_locs[0] or box == self.food_locs[1]:
                    self.del_obj(x,y)        
        if name == "swap":
            for box in range(boxes):
                x = box*2+2
                if box == self.food_locs[1]:
                    self.put_obj(Goal(reward=100, size=1.0, color='green'), x, y)
                elif box == self.food_locs[0]:
                    self.put_obj(Goal(reward=25, size=0.5, color='green'), x, y)
             
        if "blind" in name or "reveal" in name:
            splitName = name.split()
            b = self.grid.get(*self.agent_box_pos[splitName[1]])
        
            if "blind" in name:
                b.state = 1
                b.see_behind = lambda : False
            if "reveal" in name:
                b.state = 0
                b.see_behind = lambda : True
            # record whether each agent can see each food
            agent = self.instance_from_name[splitName[1]]
            for box in range(boxes):
                self.can_see[splitName[1] + str(box)] = False if "blind" in name else True

        # whenever food updates, remember locations
        if name in ["init", "swap", "replace", "reveal", "release1"] or "remove" in name or "place" in name:
            #=print(name)

            for box in range(boxes):
                x = box*2+2
                for agent in self.agents_and_puppets():
                    if self.can_see[agent+str(box)]:
                        tile = self.grid.get(x,y)
                        if hasattr(tile, "reward") and hasattr(tile, "size"):
                            #size used to distinguish treats from boxes
                            self.last_seen_reward[agent+str(box)] = tile.reward
                            #print('rew update', agent, box, tile.reward)
                        elif not self.grid.get(x,y) and self.last_seen_reward[agent+str(box)] != 0:
                            #print('0ing', box)
                            self.last_seen_reward[agent+str(box)] = 0
                        
            new_target = False
            for box in range(boxes):
                for agent in self.agents_and_puppets():
                    reward = self.last_seen_reward[agent+str(box)]
                    if (self.agent_goal[agent] != box) and (reward >= self.best_reward[agent]):
                        self.agent_goal[agent] = box
                        self.best_reward[agent] = reward
                        #print('found box', name, agent, box, reward)
                        new_target = True
                        target_agent = agent
            if new_target and target_agent != "player_0":
                a = self.instance_from_name[target_agent]
                if a.active:
                    x = self.agent_goal[target_agent]*2+2
                    #print("pathfinding to", self.agent_goal[target_agent], x, y)
                    path = pathfind(self.grid.overlapping, a.pos, (x, y), a.dir)
                    self.infos[agent]["path"] = path
                    #print('sending',path)
            


    	            
