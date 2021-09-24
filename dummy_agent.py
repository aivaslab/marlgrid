
def supervised_testing(env, agents):
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
	
	
	
