from marlgrid.envs import *
from marlgrid.envs.compfeed import *
from marlgrid.envs.knowguess import *
from marlgrid.envs.contentFB import *
from marlgrid.envs.yummyyucky import *

from marlgrid.agents import IndependentLearners, GridAgentInterface
from matplotlib import pyplot as plt
from PIL import Image
import os

def renderEnv(env, num):
	env.reset(seed=1)
	for agent, p in zip(env.agents, env.apos):
		cur_pos = agent.pos[:]
		cur_cell = env.grid.get(*cur_pos)
		fwd_pos = p[0:2]

		env.grid.set(*fwd_pos, agent)
		env.grid.set(*cur_pos, None)
		agent.pos = fwd_pos

		agent.dir = p[2]
	im = Image.fromarray(env.render(mode=''))
	im.save(os.path.join('images', type(env).__name__ + str(num) + '_im.png'))

agents1 = IndependentLearners(
	GridAgentInterface(observation_style='rich', view_size=13),
)

agents2 = IndependentLearners(
	GridAgentInterface(observation_style='rich', view_size=13),
	GridAgentInterface(observation_style='rich', view_size=13),
)

agents3 = IndependentLearners(
	GridAgentInterface(observation_style='rich', view_size=13),
	GridAgentInterface(observation_style='rich', view_size=13),
	GridAgentInterface(observation_style='rich', view_size=13),
)

envs = [KnowGuessEnv,
		KnowGuessEnv2,
		KnowGuessEnv2, #same side
		KnowGuessEnv2, #same side, one backwards
		KnowGuessEnv3, #visor
		CompFeedEnv,
		CompFeedEnv2,
		ContentFBEnv,
		ContentFBEnv2
		]

agent_pos = [[(7,3,1),(7,11,3)],
		[(4,3,1),(4,11,3), (11, 7,2)],
		[(3,11,3),(5,11,3), (11, 7,2)],
		[(3,11,3),(5,11,1), (11, 7,2)],
		[(4,3,1),(4,11,3), (11, 7,2)],
		[(2,7,0),(12,7,2)],
		[(2,7,0),(12,5,2),(12,9,2)],
		[(0,0,0),(0,1,1)],
		[(0,0,0),(0,1,1)],
		] #outdated, but duplicates of knowguess2 need ways of inputting pos
num = 0
curname = 'unnamed'
for e in envs:
	if len(e.apos) == 1:
		aList = agents1
	elif len(e.apos) == 2:
		aList = agents2
	else:
		aList = agents3
	
	env = e(agents=aList.agents, grid_size=15)
	'''except:
		try: 
			env = e(agents=aList.agents, grid_size=15, clutter_density=0.2)
		except BaseException as b:
			print(b)'''

	if type(env).__name__ != curname:
		num = 0
		curname = type(env).__name__

	print('rendering', type(env).__name__, num)
	renderEnv(env, num)
	num += 1