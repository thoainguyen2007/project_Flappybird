import gym
import pickle
from dqn_func import create_model, dqn_train
from qlearning_func import FlappyBirdCustom


env = FlappyBirdCustom(gym.make('FlappyBird-v0'), rounding = 10)
env.reset()

dqn=create_model(env)


# with open("dqn", "rb") as f:
#   dqn = pickle.load(f)

dqn_train(dqn, epsilon=1, episodes=1000,epsilon_min=.2, epsilon_decay_rate=.9999, env=env, max_steps=1000,gamma=1, alpha=.9)

##Lưu model
with open("dqn_model", "wb") as f:
  pickle.dump(dqn, f)
