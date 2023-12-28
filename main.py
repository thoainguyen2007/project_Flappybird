import numpy as np
import gym
from func import train
import gym
import gym_ple
from gym_ple import PLEEnv
# disable PLEE warning
# TODO: install the specific version of PLEE to silence warning
import warnings
warnings.filterwarnings('ignore')
class FlappyBirdCustom(gym.Wrapper):
    def __init__(self, env, rounding = 10):
      super().__init__(env)
      self.rounding = rounding

    def _discretize(self, value):
      return self.rounding * int(value / self.rounding)

    def step(self, action):
      '''
      Hàm để tính toán và trả ra custom_next_state & custom_reward
      '''
      # Reward và Internal State của môi trường
      _, reward, terminal, _ = self.env.step(action)

      # custom reward
      if reward >= 1: # nhảy qua được ống
        custom_reward = 5
      elif terminal is False:
        custom_reward = 0.5 # sống sót sau mỗi frame
      else:
        custom_reward = -1000 # gameover

      # Do thực hiện step -> ta gọi là custom_next_state
      custom_next_state = self.get_custom_state()

      # return tuple
      return custom_next_state, custom_reward, terminal

    def get_custom_state(self):
      internal_state_dict = self.env.game_state.getGameState()

      # Tính toán distance theo trục x và y
      hor_dist = internal_state_dict['next_pipe_dist_to_player']
      ver_dist = internal_state_dict['next_pipe_bottom_y'] - internal_state_dict['player_y']
      # disretize distance
      hor_dist = self._discretize(hor_dist)
      ver_dist = self._discretize(ver_dist)
      # tính toán player đang nhảy hay rơi dựa theo velocity
      is_up = 1
      if internal_state_dict['player_vel'] >= 0:
        is_up = 0
      # custom_state cho defaultdict
      custom_state = is_up,hor_dist,ver_dist

      return custom_state
env = FlappyBirdCustom(gym.make('FlappyBird-v0'), rounding = 10)
env.reset()
# define action constant
ACTION_FLAP = 0
ACTION_STAY = 1
from collections import defaultdict
counter=defaultdict(float)
q_values = defaultdict(float)
steps = train(q_values, episodes=500, env=env,counters=counter, max_steps=1000,gamma=1, alpha=.9)



import pickle

with open("new_q_values", "wb") as f:
  pickle.dump(q_values, f)


