import numpy as np
import tensorflow as tf

"""
Credits: DiagGaussian class adapted from https://github.com/openai/baselines/blob/master/baselines/common/distributions.py
"""
class DiagGaussian():
    def __init__(self, mean, logstd):
        self.set_param(mean, logstd)

    def set_param(self, mean, logstd): # Used so that we don't need to make a new DiagGaussian object every time mu and log_sigma changes
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)

    def mode(self):
        return self.mean

    def logp(self, x):
        neg_logp = 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) \
                 + 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[-1], dtype=tf.float32) \
                 + tf.reduce_sum(self.logstd, axis=-1)
        return -tf.expand_dims(neg_logp, axis=-1) # Expand dims to get correct shape

    def entropy(self):
        return tf.expand_dims(tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1), axis=-1) # Expand dims to get correct shape

    def sample(self):
        return self.mean + self.std * tf.random.normal(tf.shape(self.mean))