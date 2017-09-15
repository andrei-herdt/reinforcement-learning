#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy




# tf Graph Input

def nn(input):

    with tf.name_scope('layer1'):
        W0 = tf.Variable(tf.truncated_normal([11, 30], stddev=0.1))
        b0 = tf.Variable(tf.constant(value=0.1, shape=[30]))
        h0 = tf.matmul([input], W0) + b0

    with tf.name_scope('layer1'):
        W1 = tf.Variable(tf.truncated_normal([30, 30], stddev=0.1))
        b1 = tf.Variable(tf.constant(value=0.1, shape=[30]))
        h1 = tf.matmul(h0, W1) + b1

    with tf.name_scope('layer1'):
        W2 = tf.Variable(tf.truncated_normal([30, 3], stddev=0.1))
        b2 = tf.Variable(tf.constant(value=0.1, shape=[3]))
        h2 = tf.matmul(h1, W2) + b2

    return h2



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')


    #SET UP tensorflow
    input = tf.placeholder(tf.float32, [11])
    label = tf.placeholder(tf.float32, [3])

    pred = nn(input)

    cost = tf.losses.absolute_difference(label, pred[0])

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)


    with tf.Session() as sess:
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)


                actions.append(action)


                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

        observations = np.array(observations)

        print(observations.shape)
        actions = np.array(actions)

        for k in range(30):

            for i in range(observations.shape[0]):

                _, cost_, pred_action = sess.run([train_step, cost, pred], feed_dict={input: observations[i], label: actions[i][0]})

                print("obs:", observations[i])

                print("action:", actions[i])

                print("pred_action:", pred_action)
                print("cost: ", cost_)


            print('iter', i)
            obs = env.reset()
            done = False

            while not done:
                pred_action = sess.run(pred, feed_dict={input: obs})

                obs, r, done, _ = env.step(pred_action)
                print("r:", r)
                if args.render:
                    env.render()



if __name__ == '__main__':
    main()
