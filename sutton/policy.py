import numpy as np

class Policy:
    def __init__(self):
        self.values = np.zeros((4,4))
        self.states = []
        for i in range(4):
            for j in range(4):
                self.states.append((i,j))

    def render(self):
        for i in range(4):
            for j in range(4):
                print(self.values[i,j], end=" ")
            print()

    def eval_policy(self, env):
        gamma = 1
        # while Delta > eps:
        # Iterate over all states
        for state in self.states:
            if not env.isgoal(state):
                Vs = self.values[state[0], state[1]]
                env.state = state
                # Iterate over all actions of policy
                for i in range(4):
                    action_prob = 0.25
                    ret = env.step(i)

                    # Probability outside grid is 0, otherwise 1
                    prob = 0
                    next_state = ret[0]
                    r = ret[1]
                    if env.isinside(next_state):
                        Vs_n = self.values[next_state]
                        prob = 1
                        # Apply Belman
                        self.values[state] += action_prob*prob*(r + gamma*Vs_n)
