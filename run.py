from marlgrid.envs.cluttered import ClutteredMultiGrid
from marlgrid.agents import IndependentLearners, TestRLAgent, GridAgentInterface


def rl_loop(env, agents, episodes, rendering=False):
	for i_episode in range(N_episodes):

		obs_array = env.reset()

		with agents.episode():

			episode_over = False

			while not episode_over:
				if rendering:
					env.render()

				# Get an array with actions for each agent.
				action_array = agents.action_step(obs_array)

				# Step the multi-agent environment
				next_obs_array, reward_array, done, _ = env.step(action_array)

				# Save the transition data to replay buffers, if necessary
				agents.save_step(obs_array, action_array, next_obs_array, reward_array, done)

				obs_array = next_obs_array

				episode_over = done
				# or if "done" is per-agent:
				episode_over = all(done) # or any(done)


if __name__ == '__main__':

	#assert torch.backends.cudnn.is_acceptable(torch.cuda.FloatTensor([1.]))
	agents = IndependentLearners(TestRLAgent(observation_style='image'))
	env = ClutteredMultiGrid(agents, grid_size=15, clutter_density=0.1)