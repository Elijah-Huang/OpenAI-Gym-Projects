import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from gym.wrappers.monitoring.video_recorder import VideoRecorder

# loading env
environment_name = 'CartPole-v0'
env = gym.make(environment_name)

# getting to understand the env
'''
episodes = 5
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score += reward
    print(f'Episode:{episode} Score: {score}')
env.close()

print(env.action_space) # -> actions you can take represented by numbers
print(env.observation_space) # -> observation space represented by numbers
'''

# making and training the model
log_path = os.path.join('Training', 'Logs') # -> Training\\Logs
env = gym.make(environment_name)
env = DummyVecEnv([lambda: env]) # list of a function that just returns the env
'''model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path) # Model is agent we are training on env
# MlpPolicy -> basic neural network | verbose = 1 -> log out results | tensorboard_log for log path

model.learn(total_timesteps=25000)
'''
# of course you would want to save and reload the model so you don't need to retrain it everytime/
# so you can keep training it sustainably
# saving and reloading model
PPO_path = os.path.join('Training', 'Models', 'PPO_Cartpole')

# model.save(PPO_path) # saving
model = PPO.load(PPO_path,env=env) # reloading

# evaluating the model
'''print(evaluate_policy(model, env, n_eval_episodes = 2, render= True))
# -> average reward per episode, std of reward -> (200.0, 0.0) '''

# for recording video
video_recorder = VideoRecorder(env=env, path=os.path.join('Videos','CartPole.mp4'), enabled=True)

# evaluating the model without evaluate_policy
episodes = 1
for episode in range(1, episodes+1):
    observation = env.reset() # current observations(state)
    done = False
    score = 0

    while not done:
        env.render()
        video_recorder.capture_frame()
        action, state = model.predict(observation)
        observation, reward, done, info = env.step(action) # takes action then returns observation + stuff
        score += reward
    print(f'Episode:{episode} Score: {score}')
video_recorder.close()

# to check the logs(which can show how your model is training)
# go to cmd prompt and do:
" tensorboard --logdir='training_log_path'"
# take a look at -> Average Award, Average Episode Length


# callbacks
# to stop training after we hit some reward threshold
'''
ppo_cartpole_callback_path = os.path.join('Training', 'Models')
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200,verbose=1)
eval_callback = EvalCallback(env,
                             callback_on_new_best=stop_callback,
                             eval_freq = 10**4,
                             best_model_save_path=ppo_cartpole_callback_path,
                             verbose=1)

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path) # same as before
# model.learn(total_timesteps=2*10**4, callback = eval_callback) # now in the .learn we pass in parameter


# different policies
# this is the same as changing the architecture of your neural network
new_architecture = [{'pi':[1<<1,1<<1,1<<1,1<<1], 'vf':[1<<1,1<<1,1<<1,1<<1]}] # Custom actor (pi) and value function (vf) networks 4 layers of size 2
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path, policy_kwargs={'net_arch':new_architecture}) # changing structure of NN
model.learn(total_timesteps=2*10**4, callback = eval_callback) # now in the .learn we pass in parameter
'''