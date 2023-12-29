import matplotlib.pyplot as plt
import numpy as np
import cv2
ACTION_FLAP=0
ACTION_STAY=1

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
    

def get_optimal_action(q_values, state,env):
  q=[q_values[(state,action)] for action in (ACTION_FLAP, ACTION_STAY)]
  if q[0]==q[1]:
    return env.action_space.sample()
  return np.argmax(q)


def epsilon_greedy(q_values, state,env, epsilon):
  if np.random.rand()<epsilon:
    return env.action_space.sample()
  return get_optimal_action(q_values,state,env)


def greedy(q_values, state,counters,env):
  if counters[(state,ACTION_FLAP)]==3 and counters[(state,ACTION_STAY)]==3:
    return get_optimal_action(q_values,state,env)
  if counters[(state,ACTION_FLAP)]>=counters[(state,ACTION_STAY)]:
    counters[(state,ACTION_STAY)]+=1
    return ACTION_STAY
  counters[(state,ACTION_FLAP)]+=1
  return ACTION_FLAP


def update_q(q_values, state, action, next_state, reward, alpha, gamma):
  Q =q_values[(state,action)]
  q_max=max([q_values[(next_state,a)] for a in (ACTION_FLAP,ACTION_STAY)])
  q_values[(state,action)]=Q + alpha*(reward + gamma*q_max - Q)


def epsilon_train(q_values, episodes,epsilon,epsilon_min, epsilon_decay_rate,env, max_steps=1000, gamma=1, alpha=.9):
  steps = []
  pipes = []
  for i in range(episodes):
    env.reset()
    state = env.get_custom_state()
    step=0
    pipe=0
    while True:
      action = epsilon_greedy(q_values, state,env,epsilon)
      next_state, reward, terminal = env.step(action)
      update_q(q_values, state, action, next_state, reward, alpha, gamma)
      state=next_state
      step+=1
      if reward==5:
        pipe+=1
      if step==max_steps or terminal:
        break
    steps.append(step)
    pipes.append(pipe)
    if epsilon>epsilon_min:
      epsilon *=epsilon_decay_rate

    if (i+1)%100==0:
      print(f' episodes  {i+1} - epsilon {epsilon}')
      print(f' step   :     {np.mean(step)}')
      print(f' pipe   :    {np.mean(pipe)}')

  return steps


def new_train(q_values, episodes,env,counters, max_steps=1000, gamma=1, alpha=.9):
  steps = []
  pipes = []
  for i in range(episodes):
    env.reset()
    state = env.get_custom_state()
    step=0
    pipe=0
    while True:
      action = greedy(q_values, state, counters,env)
      next_state, reward, terminal = env.step(action)
      update_q(q_values, state, action, next_state, reward, alpha, gamma)
      state=next_state
      step+=1
      if reward==5:
        pipe+=1
      if step==max_steps or terminal:
        break
    steps.append(step)
    pipes.append(pipe)

    if (i+1)%100==0:
      print(f' episodes  {i+1}')
      print(f' step   :     {np.mean(step)}')
      print(f' pipe   :    {np.mean(pipe)}')

  return steps



def test(q_values, episodes,env):
  history = {}
  for i in range(episodes):
    env.reset()                     # bắt đầu 1 episode
    state = env.get_custom_state()  # lấy state đầu tiên

    frames = []
    score = 0
    while True:
      # hàm lấy random action từ môi trường
      action = get_optimal_action(q_values,state,env)
      # step
      state, reward, terminal = env.step(action)
      # render
      frame = env.render(mode='rgb_array')
      frames.append(frame)
      if reward ==5 :
        score+=1
      if terminal:
        history[i]={"score":score, "frames": frames}
        print(f'episodes {i} : score {score}')
        break

    for i in frames:
        cv2.imshow("image",i)
        cv2.waitKey(50)
    cv2.destroyAllWindows()
    
  return history


def visualize(steps_,pipes_):
  steps_=np.array(steps_)
  pipes_=np.array(pipes_)
  k=np.array([i for i in range(len(steps_))])
  plt.figure(figsize=(10, 5))

  plt.subplot(1,2,1)
  plt.title("Số step từng episode")
  plt.xlabel("Step")
  plt.ylabel("y1")
  plt.plot(k,steps_)

  plt.subplot(1,2,2)
  plt.title("Số pipe từng episode")
  plt.xlabel("Episode")
  plt.ylabel("Pipe")
  plt.plot(k, pipes_)

  plt.show()
