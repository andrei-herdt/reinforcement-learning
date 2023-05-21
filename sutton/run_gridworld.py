import gridworld as gw
import policy as pol

env = gw.GridWorldEnv()
pl = pol.Policy()

env.render()
pl.render()
pl.eval_policy(env)
pl.render()
