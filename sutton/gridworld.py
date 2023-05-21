import numpy as np

class GridWorldEnv:
    def __init__(self):
        self.grid = np.zeros((4, 4))
        self.start_state = (0, 0)
        self.goal_states = [(3, 3), (0,0)]
        self.state = self.start_state

    def reset(self):
        self.state = self.start_state

    def step(self, action):
        next_state = self.state
        reward = -1
        done = False

        if action == 0:
            next_state = (next_state[0], next_state[1] - 1)
        elif action == 1:
            next_state = (next_state[0], next_state[1] + 1)
        elif action == 2:
            next_state = (next_state[0] - 1, next_state[1])
        elif action == 3:
            next_state = (next_state[0] + 1, next_state[1])

        if next_state == self.goal_states[0] or next_state == self.goal_states[1]:
            reward = 1
            done = True

        return next_state, reward, done

    def isinside(self, state):
        if state[0] < 0 or state[1] < 0 or state[0] > 3 or state[1] > 3:
            return 0
        return 1

    def isgoal(self, state):
        if state == self.goal_states[0]:
            return 1
        elif state == self.goal_states[1]:
            return 1
        else:
            return 0

    def render(self):
        for i in range(4):
            for j in range(4):
                if (i, j) == self.state:
                    print("S", end="")
                elif (i, j) == self.goal_states[0]:
                    print("G", end="")
                elif (i, j) == self.goal_states[1]:
                    print("G", end="")
                else:
                    print("O", end="")
            print()
