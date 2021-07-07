import numpy as np
from stable_baselines3.td3 import expert_buffer
import gym
from matplotlib import pyplot as plt
import highway_env
import pprint
import numpy as np
from stable_baselines3.td3 import TD3withExpert
from stable_baselines3 import TD3
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import BaseCallback

ex_obs=np.load("obs_interaction_4.npy")
ex_next_obs=np.load("next_obs_interaction_4.npy")
ex_rewards=np.load("rewards_interaction_4.npy")
ex_actions=np.load("actions_interaction_4.npy")
env=gym.make("merge-v3")
env.reset()
#plt.imshow(env.render(mode="rgb_array"))
#plt.show()
#env2=gym.make("merge-v3")
#pprint.pprint(env.config)
n_actions = 2
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.01 * np.ones(n_actions))
#model =TD3("MlpPolicy",env,learning_rate=0.00001,  gamma=0.99, buffer_size=9000, optimize_memory_usage=True, action_noise=action_noise, verbose=1 , tensorboard_log="./TD3_cartpole_tensorboard/66")
#model.learn(total_timesteps=10000,eval_env=env2,eval_freq=2000,n_eval_episodes=30,tb_log_name="./TD3_cartpole_tensorboard/19")
#model.save_replay_buffer("/home/lieben/PycharmProjects/masterarbeits3/TD3")
#model.learn(total_timesteps=40000,tb_log_name="./TD3_cartpole_tensorboard/1")
model =TD3withExpert("MlpPolicy",env,learning_rate=0.00001,  gamma=0.99, buffer_size=10000, action_noise=action_noise, verbose=1 , tensorboard_log="./TD3_cartpole_tensorboard/5",ex_actions=ex_actions,ex_rewards=ex_rewards,ex_next_obs=ex_next_obs,ex_obs=ex_obs)
model.learn(total_timesteps=60000,tb_log_name="./TD3_cartpole_tensorboard/5")

#model.learn(total_timesteps=60000,tb_log_name="./TD3_cartpole_tensorboard/27")
#model.save("TD3_test42")

#eval_env=env2,eval_freq= 2000,n_eval_episodes=30,

