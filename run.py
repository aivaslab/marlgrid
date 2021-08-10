from marlgrid.envs.cluttered import ClutteredMultiGrid
from marlgrid.agents import IndependentLearners, TestRLAgent, GridAgentInterface

agents = IndependentLearners(
    TestRLAgent(),
    TestRLAgent(),
    TestRLAgent()
)

env = ClutteredMultiGrid(agents, grid_size=15, n_clutter=10)


for i_episode in range(N_episodes):

    obs_array = env.reset()

    with agents.episode():

        episode_over = False

        while not episode_over:
            # env.render()

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