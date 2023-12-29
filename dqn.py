import gym
from func import FlappyBirdCustom
from dqn_func import create_model, dqn_train

env = FlappyBirdCustom(gym.make('FlappyBird-v0'), rounding = 10)
env.reset()

dqn=create_model(env)
dqn_train(dqn, epsilon=1, episodes=500,epsilon_min=.2, epsilon_decay_rate=.9999, env=env, max_steps=1000,gamma=1, alpha=.9)

import pickle
with open("dqn1", "wb") as f:
  pickle.dump(dqn, f)