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
	im = Image.fromarray(env.render(mode='', show_agent_views=False))
	im.save(os.path.join('images', type(env).__name__ + str(num) + '_env.png'))
	im = Image.fromarray(env.render(mode='', show_agent_views=True))
	im.save(os.path.join('images', type(env).__name__ + str(num) + '_views.png'))


#contains env class and argument for init_agents
envs = [(KnowGuessEnv, 0),
		(KnowGuessEnv2, 0),
		(KnowGuessEnv2, 1), #same side
		(KnowGuessEnv2, 2), #same side, one backwards
		(KnowGuessEnv3, 0), #visor
		(CompFeedEnv, 0),
		(CompFeedEnv2, 0),
		(ContentFBEnv, 0),
		(ContentFBEnv2, 0)
		]

num = 0
curname = 'unnamed'
for e in envs:
	
	env = e[0](grid_size=15)
	env.init_agents(e[1], {'observation_style':'rich', 'view_size':13})

	if type(env).__name__ != curname:
		num = 0
		curname = type(env).__name__

	#print('rendering1', curname, num)
	renderEnv(env, num)
	num += 1