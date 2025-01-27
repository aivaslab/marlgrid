from ..base import MultiGridEnv, MultiGrid
from ..objects import *


class ClutteredMultiGrid(MultiGridEnv):
    mission = "get to the green square"
    metadata = {}

    def __init__(self, *args, n_clutter=None, clutter_density=None, randomize_goal=False, **kwargs):
        if (n_clutter is None) == (clutter_density is None):
            raise ValueError("Must provide n_clutter xor clutter_density in environment config.")

        super().__init__(*args, **kwargs)

        if clutter_density is not None:
            self.n_clutter = int(clutter_density * (self.width-2)*(self.height-2))
        else:
            self.n_clutter = n_clutter

        self.randomize_goal = randomize_goal

        # self.reset()


    def _gen_grid(self, width, height):
        self.grid = MultiGrid((width, height))
        self.grid.wall_rect(0, 0, width, height)
        if getattr(self, 'randomize_goal', True):
            self.place_obj(Goal(color="green", reward=1), max_tries=100)
        else:
            self.put_obj(Goal(color="green", reward=1), width - 2, height - 2)
        for _ in range(getattr(self, 'n_clutter', 0)):
            self.place_obj(Wall(), max_tries=100)

        self.agent_spawn_kwargs = {}
        self.place_agents(**self.agent_spawn_kwargs)

class ClutteredPrefGrid(MultiGridEnv):
    mission = "get to the preferred square"
    metadata = {}
    colors = ['green','blue','red','purple','yellow','orange']

    def __init__(self, *args, n_clutter=None, clutter_density=None, randomize_goal=False, n_goals=3, pref_color='green', **kwargs):
        if (n_clutter is None) == (clutter_density is None):
            raise ValueError("Must provide n_clutter xor clutter_density in environment config.")

        super().__init__(*args, **kwargs)

        if clutter_density is not None:
            self.n_clutter = int(clutter_density * (self.width-2)*(self.height-2))
        else:
            self.n_clutter = n_clutter

        self.randomize_goal = randomize_goal
        self.pref_color = pref_color
        self.n_goals = n_goals

        # self.reset()


    def _gen_grid(self, width, height):
        self.grid = MultiGrid((width, height))
        self.grid.wall_rect(0, 0, width, height)
        for g in n_goals:
            if getattr(self, 'randomize_goal', True):
                self.place_obj(Goal(color=colors[g], reward=(colors[g]==self.pref_color)), max_tries=100)
            else:
                self.put_obj(Goal(color=colors[g], reward=(colors[g]==self.pref_color)), width - 2 - g, height - 2)
        for _ in range(getattr(self, 'n_clutter', 0)):
            self.place_obj(Wall(), max_tries=100)

        self.agent_spawn_kwargs = {}
        self.place_agents(**self.agent_spawn_kwargs)

class ClutteredPrefSubgoalGrid(MultiGridEnv):
    mission = "get to the subgoal (green) and then to the preferred goal (pref_color)"
    metadata = {}
    colors = ['blue','red','purple','yellow','orange']

    #TODO: Make it so getting the subgoal makes the preferred subgoal swap rewards.

    def __init__(self, *args, n_clutter=None, clutter_density=None, randomize_goal=False, n_goals=3, pref_color='green', **kwargs):
        if (n_clutter is None) == (clutter_density is None):
            raise ValueError("Must provide n_clutter xor clutter_density in environment config.")

        super().__init__(*args, **kwargs)

        if clutter_density is not None:
            self.n_clutter = int(clutter_density * (self.width-2)*(self.height-2))
        else:
            self.n_clutter = n_clutter

        self.randomize_goal = randomize_goal
        self.pref_color = pref_color
        self.n_goals = n_goals

        # self.reset()


    def _gen_grid(self, width, height):
        self.grid = MultiGrid((width, height))
        self.grid.wall_rect(0, 0, width, height)

        if getattr(self, 'randomize_goal', True):
            self.place_obj(Goal(color="green", reward=1), max_tries=100)
        else:
            self.put_obj(Goal(color="green", reward=1), width - 2, height - 2)

        for g in n_goals:
            if getattr(self, 'randomize_goal', True):
                self.place_obj(Goal(color=colors[g], reward=-1), max_tries=100)
            else:
                self.put_obj(Goal(color=colors[g], reward=-1), width - 3 - g, height - 2)
        for _ in range(getattr(self, 'n_clutter', 0)):
            self.place_obj(Wall(), max_tries=100)

        self.agent_spawn_kwargs = {}
        self.place_agents(**self.agent_spawn_kwargs)