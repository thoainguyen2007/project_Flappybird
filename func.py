import numpy as np
ACTION_FLAP=0
ACTION_STAY=1

def get_optimal_action(q_values, state,env):
  q=[q_values[(state,action)] for action in (ACTION_FLAP, ACTION_STAY)]
  if q[0]==q[1]:
    return env.action_space.sample()
  return np.argmax(q)

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


def train(q_values, episodes,env,counters, max_steps=1000, gamma=1, alpha=.9):
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
  return history