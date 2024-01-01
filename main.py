import gym
from func import epsilon_train ,new_train, FlappyBirdCustom
from collections import defaultdict
import pickle

env = FlappyBirdCustom(gym.make('FlappyBird-v0'), rounding = 10)
env.reset()


###Train and save with epsilon_greedy
# epsilon_q_values = defaultdict(float)
# steps = epsilon_train(epsilon_q_values, epsilon=1, episodes=500,epsilon_min=.001, epsilon_decay_rate=.99, env=env, max_steps=1000,gamma=1, alpha=.9)

# with open("epsilon_q_values", "wb") as f:
#   pickle.dump(epsilon_q_values, f)

###Train and save with new_greedy
counter=defaultdict(float)
new_q_values = defaultdict(float)
step=new_train(new_q_values, episodes=500,env=env,counters=counter, max_steps=1000, gamma=1, alpha=.9)

with open("new_q_values", "wb") as f:
  pickle.dump(new_q_values, f)