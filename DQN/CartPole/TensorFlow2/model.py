import numpy as np
import tensorflow as tf
from tensorflow import keras

class QNetwork(keras.Model):
    def __init__(self,action_space):
        super(QNetwork,self).__init__()
        self.action_space = action_space

        self.dense1 = keras.layers.Dense(100,activation="tanh",kernel_initializer="he_normal")
        self.dense2 = keras.layers.Dense(100,activation="tanh",kernel_initializer="he_normal")
        self.out    = keras.layers.Dense(self.action_space,kernel_initializer="he_normal")

        #optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate=4e-4,epsilon=1e-7)

    #@tf.function
    def call(self,x):
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.out(x)
        return out

    def predict(self,states):
        states = np.atleast_2d(states).astype(np.float32)
        return self(states).numpy()

    def model(self,input_shape):
        x = keras.layers.Input(shape=input_shape)
        return keras.Model(inputs=[x],outputs=self.call(x))
