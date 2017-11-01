from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os,time,sys

from utils import plot_all_complex,SimpleDataIterator


################################################################
# data and parameters
ITERATIONS = 40000
CRITIC_ITERS = 5
DATA = "Geometry"
LOSS = "Sqrt"
MODE = "wgan-gp"

X_dim = 2
Z_dim = 2
H_dim = 500

data_type = tf.float32

LAMBDA = float(sys.argv[1])

BATCH_SIZE = 256

######################################################
# define model

# real data Circular Ring
R2 = 1;
R1 = np.sqrt(0.5);
Xc = 0.5;
Yc = 0.5;

circle_angle = tf.random_uniform([BATCH_SIZE, 1],0,1,dtype=data_type)* (2*np.pi)
circle_radius = tf.sqrt(tf.random_uniform([BATCH_SIZE, 1],0,1,dtype=data_type)* (R2**2- R1**2) + R1**2)
circle_x = Xc + circle_radius*tf.cos(circle_angle);
circle_y = Yc + circle_radius*tf.sin(circle_angle);
real_data_circle = tf.concat([circle_x,circle_y],axis=1)

# real data Square
square_x = tf.random_uniform([BATCH_SIZE, 1],0,1,dtype=data_type)
square_y = tf.random_uniform([BATCH_SIZE, 1],0,1,dtype=data_type)
real_data_square = tf.concat([square_x,square_y],axis=1)

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev*2,dtype=data_type)
    
def bias_init(shape):
  initial = tf.truncated_normal(shape, stddev=1,dtype=data_type)
  return initial

def generator(z,name,scope_reuse=False):
  
    with tf.variable_scope(name) as scope:
        if scope_reuse:
            scope.reuse_variables()
        G_W1 = tf.get_variable('W1',initializer=xavier_init([Z_dim, H_dim]))
        G_b1 = tf.get_variable('b1',initializer=bias_init([H_dim]))
        
        G_W2 = tf.get_variable('W2',initializer=xavier_init([H_dim, X_dim]))
        G_b2 = tf.get_variable('b2',initializer=bias_init([X_dim]))
        
        G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
        out = tf.matmul(G_h1, G_W2) + G_b2
    return out


def discriminator(x,name,scope_reuse=False):
  
    with tf.variable_scope(name) as scope:
        if scope_reuse:
            scope.reuse_variables()
            
        D_h1_dim = 512
        D_h2_dim = 512
        D_h3_dim = 512
        
        D_W0 = tf.get_variable('W0',initializer=xavier_init([X_dim, D_h1_dim]))
        D_b0 = tf.get_variable('b0',initializer=tf.zeros(shape=[D_h1_dim]))
        
        D_W1 = tf.get_variable('W1',initializer=xavier_init([D_h1_dim, D_h2_dim]))
        D_b1 = tf.get_variable('b1',initializer=tf.zeros(shape=[D_h2_dim]))
        
        D_W2 = tf.get_variable('W2',initializer=xavier_init([D_h2_dim, D_h3_dim]))
        D_b2 = tf.get_variable('b2',initializer=tf.zeros(shape=[D_h3_dim]))
        
        D_W3 = tf.get_variable('W3',initializer=xavier_init([D_h3_dim, 1]))
        D_b3 = tf.get_variable('b3',initializer=tf.zeros(shape=[1]))
        
        D_h1 = tf.tanh(tf.matmul(x, D_W0) + D_b0)
        D_h2 = tf.tanh(tf.matmul(D_h1, D_W1) + D_b1)
        D_h3 = tf.tanh(tf.matmul(D_h2, D_W2) + D_b2)
        out = tf.matmul(D_h3, D_W3) + D_b3
    
    return out


Z = tf.random_uniform([BATCH_SIZE, Z_dim],0,1,dtype=data_type)
fake_data = generator(Z,'Generator')

D1_real = discriminator(real_data_circle,'Discriminator1')
D2_real = discriminator(real_data_square,'Discriminator2')

D1_fake = discriminator(fake_data,'Discriminator1',True)
D2_fake = discriminator(fake_data,'Discriminator2',True)

D1_loss = tf.reduce_mean(D1_fake) - tf.reduce_mean(D1_real)  
D2_loss = tf.reduce_mean(D2_fake) - tf.reduce_mean(D2_real) 

G_loss = (-tf.reduce_mean(D2_fake) ) #-tf.reduce_mean(D1_fake)  + (-tf.reduce_mean(D2_fake) )

Z_fix = tf.constant(np.random.uniform(low=0.0, high=1.0, size=(3000,Z_dim)),dtype=data_type)
Fixed_sample = generator(Z_fix,'Generator',True)

train_variables = tf.trainable_variables()
generator_variables = [v for v in train_variables if v.name.startswith("Generator")]
discriminator1_variables = [v for v in train_variables if v.name.startswith("Discriminator1")]
discriminator2_variables = [v for v in train_variables if v.name.startswith("Discriminator2")]

# WGAN gradient penalty
if MODE == 'wgan-gp':
    alpha = tf.random_uniform(shape=[BATCH_SIZE,1], minval=0.,maxval=1.)
    interpolates = alpha*real_data_circle + ((1-alpha)*fake_data)
    disc_interpolates = discriminator(interpolates,'Discriminator1',True)
    gradients = tf.gradients(disc_interpolates, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1)**2)
 
    D1_loss += LAMBDA*gradient_penalty
    
    alpha = tf.random_uniform(shape=[BATCH_SIZE,1], minval=0.,maxval=1.)
    interpolates = alpha*real_data_square + ((1-alpha)*fake_data)
    disc_interpolates = discriminator(interpolates,'Discriminator2',True)
    gradients = tf.gradients(disc_interpolates, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1)**2)
 
    D2_loss += LAMBDA*gradient_penalty
    
    disc1_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(D1_loss, var_list=discriminator1_variables)
    disc2_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(D2_loss, var_list=discriminator2_variables)
    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(G_loss, var_list=generator_variables)


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

samples_circle,samples_square = sess.run([real_data_circle,real_data_square])
fig = plt.figure()
plt.scatter(samples_circle[:,0],samples_circle[:,1])
plt.savefig('out/{}.png'
            .format('real_circle'), bbox_inches='tight')
plt.close(fig)
fig = plt.figure()
plt.scatter(samples_square[:,0],samples_square[:,1])
plt.savefig('out/{}.png'
            .format('real_square'), bbox_inches='tight')
plt.close(fig)

for it in range(ITERATIONS):
    for _ in range(CRITIC_ITERS):
        D1_loss_curr, _ = sess.run([D1_loss,disc1_train_op])
        D2_loss_curr, _ = sess.run([D2_loss,disc2_train_op])
        

    G_loss_curr, _ = sess.run( [ G_loss, gen_train_op]) 
    
    if it % 100 == 0:
        print('Iter: {}; D loss: {:.4};D loss: {:.4}; G_loss: {:.4}'
              .format(it, D1_loss_curr,D1_loss_curr, G_loss_curr))
        if it % 4000 == 0:
            samples = sess.run(Fixed_sample)
            fig = plt.figure()
            plt.scatter(samples[:,0],samples[:,1])
            plt.savefig('out/{}_{}_{}_{}.png'
                        .format(DATA,'dual_square_',LAMBDA,str(it).zfill(3)), bbox_inches='tight')
            plt.close(fig)
