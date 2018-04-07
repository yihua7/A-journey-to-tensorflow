# Auto-encoder

import tensorflow as tf
import numpy as np
import sklearn.preprocessing as prep
from tensorflow.examples.tutorials.mnist import input_data

# Xaiver initializer
def xavier_init(fan_in,fan_out,constant=1):
    low=-constant*np.sqrt(6.0/(fan_in+fan_out))
    high=constant*np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform((fan_in,fan_out),minval=low,maxval=high,dtype=tf.float32)

# Define an Denoising Autoencoder.
class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self,n_input,n_hidden,transfer_function=tf.nn.softplus,optimizer=tf.train.AdamOptimizer(),scale=0.1):
        self.n_input=n_input
        self.n_hidden=n_hidden
        self.transfer=transfer_function
        self.scale=tf.placeholder(tf.float32)
        self.training_scale=scale
        network_weights=self._initialize_weights()
        # Define the construction of network.
        self.weights=network_weights
        self.x=tf.placeholder(tf.float32,[None,self.n_input])
        self.hidden=self.transfer(tf.add(tf.matmul(self.x+scale*tf.random_normal((n_input,)),self.weights['w1']),self.weights['b1']))
        self.reconstrunction=tf.add(tf.matmul(self.hidden,self.weights['w2']),self.weights['b2'])
        # Define the cost function.
        self.cost=0.5*tf.reduce_sum(tf.pow(tf.subtract(self.reconstrunction,self.x),2.0))
        self.optimizer=optimizer.minimize(self.cost)
        init=tf.global_variables_initializer()
        self.sess=tf.Session()
        self.sess.run(init)

    # Declaring a dictionary named all_weights and return it, which contains weights and biasses.
    def _initialize_weights(self):
        all_weights=dict()
        all_weights['w1']=tf.Variable(xavier_init(self.n_input,self.n_hidden))
        all_weights['b1']=tf.Variable(tf.zeros([self.n_hidden],dtype=tf.float32))
        all_weights['w2']=tf.Variable(tf.zeros([self.n_hidden,self.n_input],dtype=tf.float32))
        all_weights['b2']=tf.Varibale(tf.zeros([self.n_input],dtype=tf.float32))
        return all_weights

    # A function which calculates the cost function and executive one training step.
    def partial_fit(self,X):
        cost,opt=self.sess.run((self.cost,self.optimizer),feed_dict={self.x:X,self.scale:self.training_scale})
        return cost

    # A funtion calculating cost function only.
    def calc_total_cost(self,X):
        return self.sess.run(self.cost,feed_dict={self.x:X,self.scale:self.training_sccale})

    # One can get the features extracted by encoder through the fuction.
    def transform(self,X):
        return self.sess.run(self.hidden,feed_dict={self.x:X,self.scale:self.training_scale})

    # One can reconstruct the data from features through the function.
    def generate(self,hidden=None):
        if hidden is None:
            hidden=np.random.normal(size-self.weights['b1'])
        return self.sess.run(self.reconstruction,feed_dict={self.hidden:hidden})


    def reconstrunction(self,X):
        return self.sess.run(self.reconstrunction,feed_dict={self.x:X,self.scale:self.training_scale})

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])