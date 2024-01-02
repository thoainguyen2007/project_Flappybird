import pickle
import gym
from qlearning_func import FlappyBirdCustom, test
# from dqn_func import dqn_test

env = FlappyBirdCustom(gym.make('FlappyBird-v0'), rounding = 10)
env.reset()

###TEST###
# with open("epsilon_q_values", "rb") as f:
#   epsilon_q_values = pickle.load(f)
# epsilon_history = test(epsilon_q_values, 5,env)


with open("new_q_values", "rb") as f:
  new_q_values = pickle.load(f)
new_history=test(new_q_values, 50,env)


# with open("dqn_model", "rb") as f:
#   dqn = pickle.load(f)
# dqn_history=dqn_test(dqn, 1, env)