import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt

import time
import os

# Q学習によるCartPole
# 方策はεグリーディ（εの値はepisode数により低下）を使う

class QTable():
	def __init__(self,num_action,gamma=0.99,alpha=0.5,num_digitized=6):
		self._Qtable = np.random.uniform(low=-1,high=1,size=(num_digitized**4,num_action))
		self.gamma = gamma
		self.alpha = alpha

	def get_action(self,state,epsilon):
		if epsilon > np.random.uniform(0,1):
			next_action = np.random.choice([0,1])
		else:
			a = np.where(self._Qtable[state]==self._Qtable[state].max())[0]
			next_action = np.random.choice(a)
		return next_action

	def update_Qtable(self,state,action,reward,next_state):
		next_maxQ=max(self._Qtable[next_state])
		self._Qtable[state,action] = (1-self.alpha)*self._Qtable[state,action] + self.alpha*(reward+self.gamma*next_maxQ)
		return self._Qtable

def digitize_state(observation,num_digitized=6):
	p,v,a,w = observation
	d = num_digitized
	pn = np.digitize(p,np.linspace(-2.4,2.4,d+1)[1:-1])
	vn = np.digitize(v,np.linspace(-3.0,3.0,d+1)[1:-1])
	an = np.digitize(a,np.linspace(-0.5,0.5,d+1)[1:-1])
	wn = np.digitize(w,np.linspace(-2.0,2.0,d+1)[1:-1])
	return pn + vn*d**1 + an*d**2 + wn*d**3

def reward_plot(data,mode,name="ql"):
	path = os.path.join(".","images",name+".png")
	# mode == 0 で全描画
	# それ以外の時はmodeずつの平均にして描画
	if mode == 0:
		x = np.array(range(0,len(data)))
		y = np.array(data)

		plt.plot(x,y,label=name)
		plt.legend()
	else:
		cnt = 0
		y = []
		tmp = []
		for a in data:
			cnt += 1
			tmp.append(a)
			if cnt==mode:
				y.append(sum(tmp)/len(tmp))
				tmp=[]
				cnt=0
		if len(tmp) != 0:y.append(sum(tmp)/len(tmp))
		plt.plot(y,label=name)
		plt.legend()

	plt.savefig(path)


def main():
	# ハイパーパラメータとか
	ENV_NAME = "CartPole-v1"
	RENDER_MODE = 0 # 1で定期的に学習を描画する
	PLOT_MODE = 0 # 1で学習してきたエピソードの報酬をpng画像にする
	MOVIE_MODE = 1 # 1で学習後にQ方策にて一度プレイしそれを動画として保存する
	SEED = 42 # 乱数のシードの設定  再現性が高くなる

	NUM_EPISODES = 1000 # 総エピソード回数
	MAX_NUMBER_OF_STEPS = 500 # 各エピソードの最大行動数 CartPole-v1では最大500ステップとなる

	GAMMA = 0.99 # 減衰率
	ALPHA = 0.5 # 学習率
	NUM_DIGITIZED = 8 # 分割数

	#乱数のシード
	np.random.seed(seed=SEED)

	env = gym.make(ENV_NAME)
	env.seed(seed=SEED)
	y = []
	tab = QTable(env.action_space.n,GAMMA,ALPHA,NUM_DIGITIZED)
	for episode in range(NUM_EPISODES+1):
		obs = env.reset()
		state = digitize_state(obs,NUM_DIGITIZED)
		episode_reward = 0
		for t in range(MAX_NUMBER_OF_STEPS):
			action = tab.get_action(state,epsilon=0.5*(1/(episode+1)))
			# action = tab.get_action(state,epsilon=0)
			obs,reward,done,ifo = env.step(action)
			if RENDER_MODE:
				if episode%10 == 0:
					env.render()
			if done and t < MAX_NUMBER_OF_STEPS-1:
				reward -= MAX_NUMBER_OF_STEPS
			next_state = digitize_state(obs,NUM_DIGITIZED)
			q_table = tab.update_Qtable(state,action,reward,next_state)
			state = next_state
			episode_reward += reward
			if episode_reward == 500:
				true_total_reward = episode_reward
			else:
				true_total_reward = episode_reward + 500
			if done:
				break
		print(f'Episode:{episode:4.0f}, ER:{true_total_reward:4.0f}')
		y.append(true_total_reward)

	if PLOT_MODE:reward_plot(y,0,"test")
	# 最後に学習後の最適プレイを１回動画にする
	if MOVIE_MODE:
		vid = wrappers.monitoring.video_recorder.VideoRecorder(env,path="./movies/test.mp4")
		obs = env.reset()
		state = digitize_state(obs,NUM_DIGITIZED)
		for t in range(MAX_NUMBER_OF_STEPS):
			vid.capture_frame()
			action = tab.get_action(state,epsilon=0)
			obs,reward,done,ifo = env.step(action)
			state = digitize_state(obs,NUM_DIGITIZED)
			if done:
				print("final reward:",t+1)
				break
	env.close()

if __name__ == "__main__":
	main()
