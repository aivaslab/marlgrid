
def create_supervised_data(env, agents, num_runs=50):

	val = []

	# the data threeple
	action_history = []
	predict_history = []
	mental_history = []
	character_history = []

	episode_history = []

	traj_history = []
	grids = []
	ep_length = env.maxtime
	filler = env.get_filler()

	obs = env.reset(setting=setting, num_visible=num_goals)

	for ep in tqdm.tqdm(range(num_runs*eps_per_run)):

		buffer_s = [np.zeros(obs[0].shape) for _ in range(env.maxtime)]
		if (ep % eps_per_run) == eps_per_run-1:
			obs = env.reset(setting=setting, num_visible=num_goals)
		else:
			obs = env.reset()
		if ep % eps_per_run == 0:
			episode_number = 0
			#clear ep_history here?
			for agent in agents:
				if not unarbitrary_prefs:
					agent.reset_prefs()
				else:
					agent.hardcode_prefs()
		
		prevact = None
		prevpos = None
		agentpos = agents[0].pos
		episode_time = 0

		while not env.done:
			if rendering and ((ep % eps_per_run) == eps_per_run-1):
				env.render()
			

			buffer_s.append(obs[0])
			actions = [agent.action(torch.FloatTensor([buffer_s[-env.maxtime:]]).cuda()),]
			
			agentpos = agents[0].pos
			thistraj = env.get_trajectory(agentpos, prevact, prevpos) 

			prevpos = agentpos
			#without agent position, thisact of none is pretty meaningless
			prevact = actions[0]

			traj_history += [thistraj, ]
			
			 #moved this to before following if
			episode_time += 1

			if ((ep % eps_per_run) == eps_per_run-1): 
				# each step in last episode
				#episode number is 3
				if visualize:
					render_path(env, ep, episode_time, vispath)
					#print(actions)

				run = np.zeros((eps_per_run, ep_length, *filler.shape))

				if eps_per_run > 1:
					run[-episode_number-1:-1] = episode_history[-episode_number:] 
				
				episode = np.zeros((ep_length, *filler.shape))
				episode[ep_length-episode_time:] = traj_history[-episode_time]
				run[-1] = episode

				shortterm = np.asarray(traj_history[-1])
				action_history += [one_hot(5, actions[0]),]

				character_history += [run,]
				mental_history += [episode,]
				predict_history += [shortterm,]
				if not env.full_test:
					break
			obs, _, _, = env.step(actions)

		# end of episode
		episode = np.zeros((ep_length, *filler.shape))
		episode[ep_length-episode_time:] = traj_history[-episode_time:]
		episode_history += [episode, ]
		episode_number += 1

	return character_history, mental_history, predict_history, action_history


def format_data_torch(data, **train_kwargs):

	char = np.asarray(data[0]).astype('float32')
	# (N, Ep, F, W, H, C) = first.shape
	#first.reshape((N, Ep, F, C, H, W))
	char = np.swapaxes(char, 3, 5)

	mental = np.asarray(data[1]).astype('float32')
	# (N, F, W, H, C) = first.shape
	#first.reshape((N, F, C, H, W))
	mental = np.swapaxes(mental, 2, 4)

	query = np.asarray(data[2][:]).astype('float32')
	# (N, W, H, C) = second.shape
	#second.reshape((N, C, H, W))
	query = np.swapaxes(query, 1, 3)
	act = np.asarray(data[3][:]).astype('int32')

	char1 = torch.Tensor(char).cuda()#[:, 0, :, :, :, :]
	mental1 = torch.Tensor(mental).cuda()
	query1 = torch.Tensor(query).cuda()#[:, 0, :, :, :]
	act1 = torch.Tensor(act).cuda()

	dataset = torch.utils.data.TensorDataset(char1, mental1, query1, act1)

	return torch.utils.data.DataLoader(dataset, **train_kwargs)

def supervised_training(env, agents, data):
	dummies = [Dummy(steps, model) for agent in agents]


class DummyAgent():
	'''
	railroads the agent for some steps,
	then switches to an alternate model.
	
	railroaded steps should be included in 
	environment's test condition,
	returned as the final value of reset()
	
	predefined strategies after the railroaded
	steps are compared with the alt model's output

	'''
	def __init__(self, railroad, strategies, model):
		self.n = -1
		self.length = len(railroad)
		self.model =  model
		self.rails = railroad
		self.strats = strategies
	
	def choose_action(self, obs):
	
		if n <= self.length:
			self.n += 1
			return self.railroad[self.n], [0 for x in self.strats]
		else:
			self.n += 1
			act = self.model.choose_action(obs)
			return act, [act == x[self.n] for x in self.strats]
			
	def reset(railroad, strategies):
		self.length = len(railroad)
		self.rails = railroad
		self.strats = strategies
	
	
	
