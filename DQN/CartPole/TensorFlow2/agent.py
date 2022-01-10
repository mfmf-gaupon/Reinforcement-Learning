import os
from gym import wrappers
import numpy as np
import tensorflow as tf

from model import QNetwork
from buffer import SimpleRepalyBuffer

CURRENTDIR = os.getcwd()

class DQNAgent:
    def __init__(self,env):
        """
        gamma: 割引率
        max_experiences: リプレイバッファのサイズ
        min_experiences: この数以上のバッファが溜まっていたら学習
        batch_size: ミニバッチのサイズ
        """
        self.env             = env
        self.max_experiences = int(2e5)
        self.min_experiences = 512
        self.batch_size      = 16
        self.gamma           = 0.95
        self.update_period   = 2
        self.target_update_period = 500
        self.action_space    = self.env.action_space.n
        self.num_actions     = 1
        self.state_shape     = self.env.observation_space.shape
        self.q               = QNetwork(self.action_space)
        self.target_q        = QNetwork(self.action_space)
        self.replay_buffer   = SimpleRepalyBuffer(state_shape=self.state_shape,
                                                  action_shape=self.num_actions,
                                                  size=self.max_experiences)
        self.global_steps    = 0
        # initialize networks
        dummy_state = self.env.reset()
        self.q(dummy_state[np.newaxis,...])
        self.target_q(dummy_state[np.newaxis,...])
        #self.target_update()

    def get_epsilon(self,steps):
        return max(0.01,1.0-0.9*steps/1000)

    def get_action(self,state,epsilon):
        if np.random.random() < epsilon:
            action = np.random.choice(self.action_space)
        else:
            action = np.argmax(self.q.predict(state))
        return action

    def train(self,n_episodes=3000):
        reward_log = []
        for episode in range(n_episodes):
            state = self.env.reset()
            total_reward = 0
            timestep=0
            done = False
            loss_log = []
            while not done:
                self.global_steps += 1
                epsilon = self.get_epsilon(self.global_steps)
                action = self.get_action(state,epsilon)
                next_state,reward,done,info = self.env.step(action)
                total_reward+=reward

                # CartPole用のreward調整
                if done and timestep!=500:
                    reward = -1.

                self.replay_buffer.push(state,action,reward,next_state,done)
                state = next_state

                if self.global_steps%self.update_period == 0:
                    loss = self.update_networks()
                    loss_log.append(loss)
                if self.global_steps%self.target_update_period == 0:
                    self.target_update()

            if len(loss_log)==0 or None in loss_log:
                loss_log=[0]

            print(f"Episode {episode+1}: {total_reward}")
            print(f"Total_steps: {self.global_steps}")
            print(f"Current epsilon: {epsilon}")
            print(f"q_loss: {sum(loss_log)/len(loss_log)}")
            print()
            reward_log.append(total_reward)
        return reward_log

    def update_networks(self):
        if len(self.replay_buffer) < self.min_experiences:
            return

        (states,actions,rewards,next_states,dones) = self.replay_buffer.get_minibatch(self.batch_size)

        loss = self.update_q_network(states,actions,rewards,next_states,dones)

        """ # TODO:
        with summary writer loss
        """

        return loss

    def compute_target_q(self,next_states,rewards,dones):
        target_actions = tf.argmax(self.target_q.predict(next_states),axis=1)
        target_actions_onehot = tf.one_hot(target_actions,self.action_space)
        target_qnetwork_values = self.target_q(next_states)
        max_target_qnetwork_values = tf.reduce_sum(target_qnetwork_values * target_actions_onehot,axis=1,keepdims=True)
        target_q_values = rewards + self.gamma * (1.-dones) * max_target_qnetwork_values
        return target_q_values

    def update_q_network(self,states,actions,rewards,next_states,dones):
        target_q_values = self.compute_target_q(next_states,rewards,dones)
        with tf.GradientTape() as tape:
            q_values = self.q(states)
            actions_onehot = tf.one_hot(actions.flatten(),self.action_space)

            q_values = tf.reduce_sum(q_values*actions_onehot,axis=1,keepdims=True)
            td_error = tf.reduce_mean(tf.square(target_q_values-q_values))

        gradients = tape.gradient(td_error,self.q.trainable_variables)
        self.q.optimizer.apply_gradients(zip(gradients,self.q.trainable_variables))

        return td_error.numpy().astype(np.float32)

    def target_update(self):
        """
        要検証
        """
        self.target_q.set_weights(self.q.get_weights())
        """
        for q_weight,target_q_weight in zip(self.q.trainable_variables,self.target_q.trainable_variables):
            target_weight.assign(q_weight)
        """

    def save(self):
        path = os.path.join(CURRENTDIR,"checkpoints","q")
        self.q.save_weights(path)

    def load(self):
        path = os.path.join(CURRENTDIR,"checkpoints","q")
        self.q.load_weights(path)

    def test_play(self,n=1):
        path = os.path.join(CURRENTDIR,"movies")
        env = wrappers.Monitor(self.env,path,force=True,video_callable=(lambda e:True))
        for i in range(n):
            state = env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.get_action(state,0)
                next_state,reward,done,info = env.step(action)
                total_reward += reward
                state = next_state
            print(f"Test: {i}")
            print(f"total_reward: {total_reward}")
            print()
