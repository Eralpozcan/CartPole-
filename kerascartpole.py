import gym
import random
import keras
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense


def create_model(state_size,action_size):
    model=Sequential()
    model.add(Dense(24,input_dim=state_size,activation='relu'))
    model.add(Dense(24,activation='relu'))
    model.add(Dense(action_size,activation='linear'))
    model.compile(loss='mse',optimizer=keras.optimizers.Adam(lr=0.001))
    return model


if __name__=="__main__":
    env=gym.make('CartPole-v1')

    state_size=env.observation_space.shape[0]
    action_size=env.action_space.n

    batch_size =32

    game_history=deque(maxlen=2000)

    model=create_model(state_size,action_size)

    epsilon=1.0
    epsilon_min=0.01
    epsilon_decay=0.995

    for i in range(1000):
        state= env.reset()

        state=state.reshape(1,-1)
       
        for j in range(500):
            env.render()
            if np.random.rand()<=epsilon:
                action=random.randrange(action_size)
            else:
                act_values = model.predict(state)
                action=np.argmax(act_values[0])
            next_state, reward, done, _ = env.step(action)
            if done:
                reward=-10
            next_state = next_state.reshape(1,-1)

            game_history.append((state,action,reward,next_state,done))

            state=next_state
   
            if done : 
                print("Episode : {}/{},score :{}, e :{:.2}"
                    .format(i, 1000 , j, epsilon))
                break
        if len(game_history) > batch_size:
            batch=random.sample(game_history, batch_size)
        
            for state,action,reward,next_state,done in batch:
                target=reward

                if not done:
                    target=(reward +0.95 * np.amax(model.predict(next_state)[0]))

                target_action= model.predict(state)
                target_action[0][action]=target
                model.fit(state,target_action,epochs=1,verbose=0)

            if epsilon > epsilon_min:
               epsilon*=epsilon_decay     

if __name__=='_main_':
 main()