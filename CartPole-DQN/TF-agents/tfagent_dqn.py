import gym
from gym import wrappers
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow import keras
import tensorflow as tf
import time

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver, dynamic_episode_driver
from tf_agents.environments import gym_wrapper,py_environment,tf_py_environment
from tf_agents.networks import network
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common

"""
ハイパーパラメータ
"""
ENV_NAME = "CartPole-v1"
RENDER_MODE = 0 # 1:描画を行う　0:行わない
RENDER_EACH_EPISODE = 10 # 何エピソード毎に描画するか
PLOT_MODE = 1 # 1:学習の報酬をグラフにする
MOVIE_MODE = 1 # 1:学習後のプレイを１回動画にする

N_STEP_UPDATE = 1
LEARNING_RATE = 1e-3 # optimizer Adam の学習率
ADAM_EPSILON = 1e-2 # optimizer Adam のイプシロン
TARGET_UPDATE_PERIOD = 100 # target networkの更新頻度
GAMMA = 0.99 # 将来報酬の割引率

MAX_LENGTH = 10**4 # リプレイバッファのサイズ数

INIT_EPISODES = 300 # 最初に何エピソード分回して経験を集めておくか

EPISODES = 500 # 学習を何エピソード分回すか

#ネットワーククラスの設定
class Qnetwork(network.Network):
	def __init__(self,observation_spec,action_spec,name="QNetwork"):
		super(Qnetwork,self).__init__(
			input_tensor_spec=observation_spec,
			state_spec=(),
			name=name,
			)
		# 選択できる行動の数
		n_action = action_spec.maximum - action_spec.minimum + 1
		# 学習に使うnnモデル
		self.model = keras.Sequential(
			[
			 	keras.layers.Dense(50,activation="tanh",kernel_initializer="he_normal"),
			 	keras.layers.Dense(50,activation="tanh",kernel_initializer="he_normal"),
			 	keras.layers.Dense(n_action),
			]
		)
	# 行動を出力する関数
	def call(self,observation,step_type=None,network_state=(),training=True):
		actions = self.model(observation,training=training)
		return actions,network_state

def main():
	# 環境の設定
	env_py = gym.make(ENV_NAME)
	if MOVIE_MODE:
		env_py = wrappers.Monitor(env_py,"./movies",force=True,video_callable=(lambda ep:ep%50==0))
	env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(env_py))
	init_env = gym.make(ENV_NAME)
	init_env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(init_env))

	# ネットワークの設定
	primary_network = Qnetwork(env.observation_spec(),env.action_spec())
	# エージェントの設定
	agent = dqn_agent.DqnAgent(
		env.time_step_spec(),
		env.action_spec(),
		q_network=primary_network,
		optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE,epsilon=ADAM_EPSILON),
		n_step_update=N_STEP_UPDATE,
		target_update_period=TARGET_UPDATE_PERIOD,
		gamma=GAMMA,
		td_errors_loss_fn=common.element_wise_squared_loss, # mse
		# td_errors_loss_fn=keras.losses.Huber(), # huber
		train_step_counter=tf.Variable(0),
		)
	agent.initialize()
	agent.train = common.function(agent.train)
	# 行動の設定
	policy = agent.collect_policy
	# データ保存(replay buffer)の設定
	replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
		data_spec=agent.collect_data_spec,
		batch_size=env.batch_size,
		max_length=MAX_LENGTH,
		)
	dataset = replay_buffer.as_dataset(
		num_parallel_calls=tf.data.experimental.AUTOTUNE,
		sample_batch_size=64,
		num_steps=N_STEP_UPDATE+1,
		).prefetch(tf.data.experimental.AUTOTUNE)
	iterator = iter(dataset)
	# ドライバの設定
	driver = dynamic_episode_driver.DynamicEpisodeDriver(
		init_env,
		policy,
		observers=[replay_buffer.add_batch],
		num_episodes=INIT_EPISODES,
		)

	# 前処理
	env.reset()
	init_env.reset()
	driver.run(maximum_iterations=10000)

	history = []
	for episode in range(EPISODES+1):
		episode_rewards = 0
		episode_average_loss = []
		policy._epsilon = 0.5*(1/(episode+1)) # ランダム行動の確率
		time_step = env.reset()

		while True:
			if RENDER_MODE:
				if episode%RENDER_EACH_EPISODE == 0:
					env_py.render()

			policy_step = policy.action(time_step)
			next_time_step = env.step(policy_step.action)

			traj = trajectory.from_transition(time_step,policy_step,next_time_step) # データの作成
			replay_buffer.add_batch(traj) # データの保存

			experience,_ = next(iterator)
			loss_info = agent.train(experience=experience)

			R = next_time_step.reward.numpy().astype("int").tolist()[0]
			episode_average_loss.append(loss_info.loss.numpy())
			episode_rewards += R #報酬の合計値の計算

			time_step = next_time_step

			if next_time_step.is_last():
				break

		history.append(episode_rewards)
		print(f'Episode:{episode:4.0f}')
		print(f'Episode Reward:{episode_rewards:3.0f}')
		print(f'Loss:{np.mean(episode_average_loss):.4f}')
		print(f'Current Epsilon:{policy._epsilon:.6f}')
		print()

	if PLOT_MODE:
		plt.plot(history)
		plt.xlabel("episodes")
		plt.ylabel("total rewards")
		path = os.path.join(".","images","dqn.png")
		plt.savefig(path)
	env.close()

if __name__ == '__main__':
	main()
