import bandits as bs

distributions = bs.create_normal_bandits(10)

# Print the parameters of the normal distributions.
for distribution in distributions:
  print(distribution)
  print(bs.sample_bandit(distribution))
