import numpy as np

def create_normal_bandits(count):
  """Creates a list of bandits that contain parameters of the normal distribution.

  Args:
    count: The number of bandits to create.

  Returns:
    A list of bandits that contain parameters of the normal distribution.
  """

  bandits = []
  for i in range(count):
    mean = np.random.rand()
    stddev = np.random.rand()
    bandits.append({
        "mean": mean,
        "stddev": stddev,
    })

  return bandits

def sample_bandit(bandit):
  sample = np.random.uniform(bandit['mean'], bandit['stddev'])
  return sample
