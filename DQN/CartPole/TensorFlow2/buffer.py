import numpy as np

class SimpleRepalyBuffer(object):
    """
    経験（遷移情報）を貯めておくクラス
    """
    def __init__(self,state_shape,action_shape,size=100000):
        self.size        = size
        self.states      = np.empty((self.size,*state_shape),dtype=np.float32)
        self.actions     = np.empty((self.size,action_shape),dtype=np.float32)
        self.rewards     = np.empty((self.size,1),dtype=np.float32)
        self.next_states = np.empty((self.size,*state_shape),dtype=np.float32)
        self.dones       = np.empty((self.size,1),dtype=np.float32)
        self.point       = 0
        self.full        = False

    def __len__(self):
        return self.size if self.full else self.point

    def push(self,state,action,reward,next_state,done):
        np.copyto(self.states[self.point],state)
        np.copyto(self.actions[self.point],action)
        np.copyto(self.rewards[self.point],reward)
        np.copyto(self.next_states[self.point],next_state)
        np.copyto(self.dones[self.point],done)

        self.point = (self.point+1) % self.size
        self.full = self.full or self.point == 0

    def get_minibatch(self,batch_size):
        idexs = np.random.randint(0,
                                  self.size if self.full else self.point,
                                  size=batch_size)

        states = self.states[idexs]
        actions = self.actions[idexs]
        rewards = self.rewards[idexs]
        next_states = self.next_states[idexs]
        dones = self.dones[idexs]

        return (states, actions, rewards, next_states, dones)
