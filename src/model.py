import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.models import Model

from src.distribution import DiagGaussian

# Better to NOT have shared network layers between actor and critic when dealing with continuous control problems
class Model_Actor_Critic(Model):
    def __init__(self, state_space_dim, action_space_dim,
                 actor_hidden_units, critic_hidden_units):
        super(Model_Actor_Critic, self).__init__()

        self.input_layer = InputLayer(input_shape=(state_space_dim,))
        
        # Actor layers
        self.actor_hidden_layers = []
        for units in actor_hidden_units:
            self.actor_hidden_layers.append(Dense(units=units, activation='relu', kernel_initializer='GlorotNormal'))
        self.mu = Dense(units=action_space_dim, activation='tanh', kernel_initializer='GlorotNormal')
        self.log_sigma = tf.Variable(initial_value=np.zeros(shape=(1, action_space_dim)), dtype=tf.float32) 
        self.trainable_weights.append(self.log_sigma) # Add log_sigma to list of trainable variables for model

        # Critic layers
        self.critic_hidden_layers = []
        for units in critic_hidden_units:
            self.critic_hidden_layers.append(Dense(units=units, activation='relu', kernel_initializer='GlorotNormal'))
        self.values = Dense(units=1, activation=None, kernel_initializer='GlorotNormal')

        self.dist = DiagGaussian(self.mu, self.log_sigma)

    # @tf.function decoration runs function in graph mode - signficantly faster than eager execution in tf 2.0
    @tf.function
    def __call__(self, states):
        states = tf.cast(states, tf.float32) # Single precision required, explict casting in __call__
        x = y = self.input_layer(states)

        for layer in self.actor_hidden_layers:
            x = layer(x)

        for layer in self.critic_hidden_layers:
            y = layer(y)
            
        return self.mu(x), self.values(y)

    def act(self, state):
        mu, value = self(state[np.newaxis, :]) # Add new axis to make tensor the correct shape (batch size, state_dim)
        self.dist.set_param(mu, self.log_sigma)

        action = self.dist.sample()
        action_logp = self.dist.logp(action)
        
        return value[0].numpy(), action[0].numpy(), action_logp[0].numpy()
    
    def evaluate_actions(self, states, actions):
        mu, values = self(states)
        self.dist.set_param(mu, self.log_sigma)

        return values, self.dist.logp(actions), self.dist.entropy()
    
    def get_value(self, state):
        _, value = self(state[np.newaxis, :]) # Add new axis to make tensor the correct shape (batch size, state_dim)

        return value[0].numpy()