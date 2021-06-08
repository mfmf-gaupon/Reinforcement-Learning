import os
from dataclasses import dataclass
from collections import deque

import gym
from gym import wrappers
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

"""
ハイパーパラメータ
"""
ENV_NAME = "CartPole-v1"
PLOT_MODE = 1 # 1:学習の報酬をグラフにする
MOVIE_MODE = 1 # 1:学習後のプレイを１回動画にする

GAMMA = 0.99
MAX_EXPERIENCES = 10**4
MIN_EXPERIENCES = 512
BATCH_SIZE = 128
EPSILON = 1e-2
LR = 1e-3

COPY_EPISODE = 1
OPTIMIZER = keras.optimizers.Adam(learning_rate=LR,epsilon=EPSILON)

EPISODES = 500

@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool

class ReplayBuffer:
	"""
	経験（遷移情報）をためておくクラス
	"""
	def __init__(self,max_len):
		self.max_len = max_len
		self.buffer = deque(maxlen=self.max_len)

	def push(self,experience,clip_reward=False):
		state,action,reward,next_state,done = experience
		exp = Experience(state,action,reward,next_state,done)
		self.buffer.append(exp)

	def get_minibatch(self,batch_size):
		n = self.len()
		indices = np.random.choice(np.arange(n),size=batch_size,replace=False)
		selected_experiences = [self.buffer[idx] for idx in indices]


		states = np.array([exp.state for exp in selected_experiences]).astype(np.float32)
		actions = np.array([exp.action for exp in selected_experiences]).astype(np.int8)
		rewards = np.vstack([exp.reward for exp in selected_experiences]).astype(np.float32)
		next_states = np.array([exp.next_state for exp in selected_experiences]).astype(np.float32)
		dones = np.vstack([exp.done for exp in selected_experiences])

		return (states,actions,rewards,next_states,dones)

	def len(self):
		return len(self.buffer)

class QNetwork(keras.Model):
	"""
	Deep Q Networkをkerasのモデルを使用して作成(ただしCNNではなくDNNですませた)
	"""
	def __init__(self,action_space):
		super(QNetwork,self).__init__()
		self.action_space = action_space

		# nnモデル
		self.dense1 = keras.layers.Dense(50,activation="tanh",kernel_initializer="he_normal")
		self.dense2 = keras.layers.Dense(50,activation="tanh",kernel_initializer="he_normal")
		self.out = keras.layers.Dense(action_space,kernel_initializer="he_normal")

	@tf.function
	def call(self,x):
		x = self.dense1(x)
		x = self.dense2(x)
		out = self.out(x)
		return out

	def predict(self,states):
		states = np.atleast_2d(states).astype(np.float32)
		return self(states).numpy()

class DQNAgent:
	def __init__(self,
				 env,
				 gamma=0.99,
				 max_experiences=1e5,
				 min_experiences =512,
				 batch_size=128,
				 optimizer=keras.optimizers.Adam(learning_rate=1e-3),
				 ):
		"""
		gamma: 割引率
		max_experiences: リプレイバッファの最大数
		min_experiences: 学習における最低経験数
		batch_size: ミニバッチ学習のサイズ
		copy_episode: 何エピソードごとにtarget networkに重みを更新するか
		optimizer: optimizer
		"""
		self.max_experiences = max_experiences
		self.min_experiences = min_experiences
		self.batch_size = batch_size
		self.env = env
		self.gamma = gamma
		self.q_network = QNetwork(self.env.action_space.n)
		self.q_network.build(input_shape=(None,4))
		self.target_qnetwork = QNetwork(self.env.action_space.n)
		self.target_qnetwork.build(input_shape=(None,4))
		self.replay_buffer = ReplayBuffer(max_len=max_experiences)
		self.optimizer = optimizer
		self.loss_fc = tf.losses.Huber()

	def get_epsilon(self,num_episode):
		"""
		εグリーティに使うεの計算
		"""
		return max(0.01,0.5*(1/(num_episode+1)))

	def get_action(self,state,epsilon):
		"""
		εグリーディにより行動を選択
		"""
		if np.random.random() < epsilon:
			action = np.random.choice(self.env.action_space.n)
		else:
			action = np.argmax(self.q_network.predict(state))
		return action

	def train(self,epsilon):
		if self.replay_buffer.len() < self.min_experiences:
			return
		(states,actions,rewards,next_states,dones) = self.replay_buffer.get_minibatch(self.batch_size)

		target_actions = tf.argmax(self.target_qnetwork.predict(next_states),axis=1)
		target_actions_onehot = tf.one_hot(target_actions,self.env.action_space.n)
		target_qnetwork_values = self.target_qnetwork(next_states)
		max_target_qnetwork_values = tf.reduce_sum(target_qnetwork_values * target_actions_onehot,axis=1,keepdims=True)
		target_q_values = rewards + self.gamma * (1-dones) * max_target_qnetwork_values

		with tf.GradientTape() as tape:
			q_values = self.q_network(states)
			actions_onehot = tf.one_hot(actions,self.env.action_space.n) # flattenいらなくね

			q_value = tf.reduce_sum(q_values*actions_onehot,axis=1,keepdims=True)
			loss = tf.reduce_mean(tf.square(target_q_values-q_value))

		gradients = tape.gradient(loss,self.q_network.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients,self.q_network.trainable_variables))

		return loss.numpy().astype(np.float32)


	def init_play(self,num_episodes):
		"""
		はじめに学習はせずランダムな行動で経験をnum_episodes数分貯める
		"""
		for _ in range(num_episodes):
			state = self.env.reset()
			done = False
			step = 0
			while not done:
				action = self.get_action(state,1)
				next_state,reward,done,info = self.env.step(action)

				if done and step!=500:
					# cartpole 用にrewardを調整
					reward = -1
				self.add_experience((state,action,reward,next_state,done))
				state = next_state
				step += 1

	def add_experience(self,experience):
		self.replay_buffer.push(experience)

	def target_update(self):
		self.target_qnetwork.set_weights(self.q_network.get_weights())

def main():
	"""
	メインルーチン
	"""
	reward_log = []
	loss_log = []
	env = gym.make(ENV_NAME)

	agent = DQNAgent(env=env,gamma=GAMMA,max_experiences=MAX_EXPERIENCES,
				 	min_experiences=MIN_EXPERIENCES,batch_size=BATCH_SIZE,
				 	optimizer=OPTIMIZER)

	agent.init_play(100)
	agent.target_update()
	if MOVIE_MODE:
		agent.env = wrappers.Monitor(env,"./movies",force=True,video_callable=(lambda ep:ep%50==0))
	# main loop
	for n in range(EPISODES+1):
		state = agent.env.reset()
		epsilon = agent.get_epsilon(n)
		total_reward = 0
		loss_tmp = []
		step = 0
		done = False
		while not done:
			action = agent.get_action(state,epsilon)
			next_state,reward,done,info = agent.env.step(action)
			total_reward += reward

			if done and step!=500:
				# cartpole 用にrewardを調整
				reward = -1

			agent.add_experience((state,action,reward,next_state,done))
			loss = agent.train(epsilon)
			loss_tmp.append(loss if loss != None else 0)
			state = next_state
			step += 1
		if n%COPY_EPISODE==0:
			agent.target_update()

		average_loss = sum(loss_tmp)/len(loss_tmp)
		print(f"Episode {n}:{total_reward}")
		print(f"Average Loss:{average_loss}")
		print(f"Current epsilon:{epsilon}")
		print()
		reward_log.append(total_reward)
		loss_log.append(average_loss)

	#plt.plot(range(len(loss_log)),loss_log)
	#plt.show()

	plt.plot(range(len(reward_log)),reward_log)
	plt.xlabel("episodes")
	plt.ylabel("Total Reward")
	if PLOT_MODE:
		path = os.path.join(".","images","dqn.png")
		plt.savefig(path)
	else:
		plt.show()

if __name__ == "__main__":
	main()
