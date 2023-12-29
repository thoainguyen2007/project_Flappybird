import pickle
# with open("epsilon_q_values", "rb") as f:
#   epsilon_q_values = pickle.load(f)

# with open("new_q_values", "rb") as f:
#   new_q_values = pickle.load(f)

with open("dqn", "rb") as f:
  dqn = pickle.load(f)

import gym
from func import FlappyBirdCustom
from dqn_func import dqn_test
env = FlappyBirdCustom(gym.make('FlappyBird-v0'), rounding = 10)
env.reset()


###TEST###
# epsilon_history = test(epsilon_q_values, 1,env)
# new_history=test(new_q_values, 1,env)
dqn_history=dqn_test(dqn, 1, env)



