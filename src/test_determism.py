import unittest

def neighbors(env, grid):
    neigh = []

    actions = env.actions
    agents = env.agents

    for agent in agents:
        for action in actions:
            env.grid = grid
            act = {agent: action}
            env.step(act)
            neigh.append(env.grid)

    return neigh

class TestEnvironments(unittest.TestCase):

    def test_terminates(self, env, seed=0):
        # perform a bfs on an environment's search space to make sure it is possible
        visited = []
        queue = []
        visited.append(hash(env.grid))
        queue.append(env.grid)

        while queue:
            m = queue.pop(0)
            for neighbor in neighbors(env, env.grid):
                if env.done:
                    pass
                if hash(neighbor) not in visited:
                    visited.append(hash(neighbor))
                    queue.append(neighbor)

        raise "terminal node not found"

    def test_deterministic_reset(self, env, seed):
        # generate multiple sequences of rooms and tests that they have identical grids

        env.seed = seed
        room1 = hash(env.grid)
        env.reset()
        room2 = hash(env.grid)
        env.reset()

        assert room1 != room2

        env.seed = seed
        room3 = hash(env.grid)
        env.reset()
        room4 = hash(env.grid)
        env.reset()

        assert room3 != room4
        assert room1 == room3 and room2 == room4

        pass

    def test_sb3_conversion(self):
        # test whether a given environment converts to a vectorized environment,
        # can do basics (run, step), has callbacks (reward)
        # includes sb3's tests
        pass

    def test_wrappers(self):
        # test whether unwrap properly unwraps a vectorized environment

    def test_observation(self):
        # make sure observation size matches model

if __name__ == '__main__':
    unittest.main()