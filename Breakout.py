import gym
from stable_baselines3 import A2C, DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import time

# set up env
# go download Roms.rar and extract it in the current directory
# $ python -m atari_py.import_roms "C:\Users\elija\Desktop\CODING\Reinforcement Learning\Introduction (Open AI)\ROMS"
environment_name = 'Breakout-v0'
env = gym.make(environment_name)

'''
# take a look at how the env looks like
print(env.reset())
print(env.action_space) # -> discrete 4
print(env.observation_space) # box -> , (..., (210, 160, 3), uint8) -> seems to be an image

episodes = 3
for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        obs, reward, done, info = env.step(env.action_space.sample()) # random action
        score += reward

    print(f'Episode:{episode} Score:{score}')
'''

# making environment
'''
env = make_atari_env('Breakout-v0', n_envs=8, seed=0) # helper that creates wrapped atari environment
env = VecFrameStack(env, n_stack=8) # stacks environments together
# vectorized environment -> trains model on 8 different environments at the same time per time step
'''
env = make_atari_env('Breakout-v0', seed=0) # helper that creates wrapped atari environment
env = VecFrameStack(env, n_stack=1) # stacks environments together

# making and training model
'''
log_path = os.path.join('Training','Logs')
model = A2C('CnnPolicy',env, verbose=1, tensorboard_log=log_path) # verbose -> will use logs
# tensorboard_log because we want a tensorboard log

model.learn(total_timesteps=10**5)
'''

# training with DQN instead:
''''
log_path = os.path.join('Training','Logs')
model = DQN('CnnPolicy',env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=10**6)
'''

# saving
'''
a2c_breakout_path = os.path.join('Training','Models','A2C Breakout')
model.save(a2c_breakout_path)
'''
dqn_breakout_path = os.path.join('Training','Models','DQN Breakout')
'''
dqn_breakout_path = os.path.join('Training','Models','DQN Breakout')
model.save(dqn_breakout_path)
'''

# testing
env = make_atari_env('Breakout-v0', n_envs=1, seed=0) # singular environment
# env = VecFrameStack(env, n_stack=8) # leveraging vectorized model on singular environment
env = VecFrameStack(env, n_stack=1)

# reloading
'''model = A2C.load(a2c_breakout_path, env)'''
model = DQN.load(dqn_breakout_path, env)

# evaluating the model
'''print(evaluate_policy(model=model, env=env, n_eval_episodes=5, render=True,))
#-> avg reward per episode, std deviation -> (5.8, 2.2271057451320084)'''

# for recording video
video_recorder = VideoRecorder(env=env, path=os.path.join('Videos-Images','Breakout DQN Deterministic.mp4'), enabled=True)

# evaluating the model without evaluate_policy
episodes = 1
for episode in range(1, episodes+1):
    observation = env.reset() # current observations(state)
    lives = 5
    score = 0

    while lives:
        #print(lives)

        for i in range(3): # slow down video
            env.render()
            video_recorder.capture_frame()

        action, state = model.predict(observation, deterministic=True) # trained agent
        # action = [env.action_space.sample()] # random agent
        observation, reward, done, info = env.step(action) # takes action then returns observation + stuff

        lives -= done # turns out done is 1 if you lose a life and not if it's actually done
        score += reward

    print(f'Episode:{episode} Score: {score}')
video_recorder.close()