import tensorflow as tf
import gym
#from keras.models import Sequential
#from keras.layers import Dense, Lambda

# Obtain data through simulation with OpenAI
def rollout(env, policy_fun, render):
    max_steps = env.spec.timestep_limit

    observations = []
    actions = []
    total_rewards = []
    steps_numbers = []
    obs = env.reset()
    one = False
    totalr = 0.
    steps = 0
    done = False
    while not done:
        action = policy_fun(obs[None,:])
        observations.append(obs)

        actions.append(action)
        obs, r, done, _ = env.step(action)
        totalr += r
        steps += 1
        if render:
            env.render()
        if steps >= max_steps:
            break

    return observations, actions, totalr, steps


def neural_network(input, dimensions):

    with tf.name_scope('layer1'):
        W0 = tf.Variable(tf.truncated_normal([dimensions[0], 30], stddev=0.1))
        b0 = tf.Variable(tf.constant(value=0.1, shape=[30]))
        h0 = tf.matmul(input, W0) + b0

    with tf.name_scope('layer2'):
        W1 = tf.Variable(tf.truncated_normal([30, 30], stddev=0.1))
        b1 = tf.Variable(tf.constant(value=0.1, shape=[30]))
        h1 = tf.matmul(h0, W1) + b1

    with tf.name_scope('layer3'):
        W2 = tf.Variable(tf.truncated_normal([30, dimensions[1]], stddev=0.1))
        b2 = tf.Variable(tf.constant(value=0.1, shape=[dimensions[1]]))
        h2 = tf.matmul(h1, W2) + b2

    return h2


#ef baseline_model():
#   model = Sequential()
#   model.add(Dense(input_dim/2, input_dim=input_dim, init='normal', activation='relu'))
#   model.add(Dense(output_dim, init='normal'))

#   model.compile(loss='mse', optimizer='adam', metrics=['mae'])
#   return model
