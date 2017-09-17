import tensorflow as tf
import gym


env = gym.make(envname)
max_steps = env.spec.timestep_limit


# Obtain data through simulation with OpenAI
def rollout():

    observations = []
    actions = []
    obs = env.reset()
    one = False
    totalr = 0.
    steps = 0
    done = False
    while not done:
        action = policy_fn(obs[None,:])
        observations.append(obs.reshape(1,11))

        actions.append(action)
        obs, r, done, _ = env.step(action)
        totalr += r
        steps += 1
        if render:
            env.render()
        if steps >= max_steps:
            break
    returns.append(totalr)

    return observations, actions


def neural_network(input):

    with tf.name_scope('layer1'):
        W0 = tf.Variable(tf.truncated_normal([11, 30], stddev=0.1))
        b0 = tf.Variable(tf.constant(value=0.1, shape=[30]))
        h0 = tf.matmul(input, W0) + b0

    with tf.name_scope('layer2'):
        W1 = tf.Variable(tf.truncated_normal([30, 30], stddev=0.1))
        b1 = tf.Variable(tf.constant(value=0.1, shape=[30]))
        h1 = tf.matmul(h0, W1) + b1

    with tf.name_scope('layer3'):
        W2 = tf.Variable(tf.truncated_normal([30, 3], stddev=0.1))
        b2 = tf.Variable(tf.constant(value=0.1, shape=[3]))
        h2 = tf.matmul(h1, W2) + b2

    return h2