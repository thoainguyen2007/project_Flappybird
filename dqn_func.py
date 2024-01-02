from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from qlearning_func import visualize

def create_model(env):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(np.array(env.get_custom_state()).shape)))
    model.add(Dense(2))
    model.compile(loss='mse', optimizer='adam', metrics='accuracy')
    return model

def get_optimal_action(dqn, state,env):
  state=np.array([state])
  q=dqn.predict(state,verbose=0)[0]
  if q[0]==q[1]:
    return env.action_space.sample()
  return np.argmax(q)


def epsilon_greedy(dqn, state,env, epsilon):
  if np.random.rand()<epsilon:
    return env.action_space.sample()
  return get_optimal_action(dqn,state,env)

def fit(dqn,memory, gamma):
  states = np.array([state for state,_,_,_,_ in memory])
  next_states = np.array([next_state for _,_,_,next_state,_ in memory])
  qs = dqn.predict(states, verbose=0)
  next_qs = dqn.predict(next_states, verbose=0)

  for i, (state,action,reward,next_state,terminal) in enumerate(memory):
    qs[i][action]=reward+(1-terminal)*gamma*np.max(next_qs[i])

  dqn.fit(states, qs, verbose=0)


def dqn_train(dqn, episodes,epsilon,epsilon_min, epsilon_decay_rate,env, max_steps=1000, gamma=1, alpha=.9):
  steps = []
  pipes = []
  for i in range(episodes):
    env.reset()
    state = env.get_custom_state()
    step=0
    pipe=0
    memory=[]
    while True:
      action = epsilon_greedy(dqn, state,env,epsilon)
      next_state, reward, terminal = env.step(action)
      memory.append((state,action,reward,next_state,terminal))
      state=next_state
      step+=1
      if reward==5:
        pipe+=1
      if step==max_steps or terminal:
        break
    fit(dqn,memory,gamma)
    
    steps.append(step)
    pipes.append(pipe)

    if epsilon>epsilon_min:
      epsilon *=epsilon_decay_rate

    if (i+1)%10==0:
      print(f' episodes   {i-8} - {i+1} ')
      print(f' step trung bình :  {np.mean(steps[-10:])}')
      print(f' max step        :  {np.max(steps[-10:])}')
      print(f' max pipe        :  {np.max(pipes[-10:])}')

  visualize(steps, pipes)
import cv2
def dqn_test(dqn, episodes,env):
  history = {}
  for i in range(episodes):
    env.reset()                     # bắt đầu 1 episode
    state = env.get_custom_state()  # lấy state đầu tiên

    frames = []
    score = 0
    while True:
      # hàm lấy random action từ môi trường
      action = get_optimal_action(dqn,state,env)
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

