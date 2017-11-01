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

ITERATIONS = 20000
DATA = "Geometry"
LOSS = "Sqrt"

X_dim = 2
Z_dim = 2
H_dim = 500

data_type = tf.float32

config = sys.argv[1]
if config=='10': 
    L = 10
    BATCH_SIZE = 200
    epsilon_list = [100.,10.,10.,1., 1.]
elif config=='100':
    L = 100
    BATCH_SIZE = 200
    epsilon_list = [100.,10.,10.,1., 0.1]
elif config=='300':
    L = 300
    BATCH_SIZE = 10
    epsilon_list = [100.,10.,10.,1., 0.1]



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
    
epsilon = tf.placeholder(data_type, shape=())

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


Z = tf.random_uniform([BATCH_SIZE, Z_dim],0,1,dtype=data_type)
fake_data = generator(Z,'Generator')

residual =  tf.expand_dims(fake_data,axis=1) -  tf.expand_dims(real_data_circle,axis=0)        
C_mn = tf.norm(residual, ord='euclidean', axis=-1, keep_dims=False)                                                                                                                                                                             
K_mn = tf.exp(-C_mn/epsilon)

b = tf.ones([tf.shape(real_data_circle)[0],1],dtype=data_type)
for i in range(L): #use while loop or scan
    a = 1/tf.matmul(K_mn,b)
    b = 1/tf.matmul(tf.transpose(K_mn),a)

G_loss1 = tf.reduce_sum(  tf.matmul(K_mn*C_mn,b) * a)   # * element-wise

residual =  tf.expand_dims(fake_data,axis=1) -  tf.expand_dims(real_data_square,axis=0)        
C_mn = tf.norm(residual, ord='euclidean', axis=-1, keep_dims=False)                                                                                                                                                                             
K_mn = tf.exp(-C_mn/epsilon)

b = tf.ones([tf.shape(real_data_square)[0],1],dtype=data_type)
for i in range(L): #use while loop or scan
    a = 1/tf.matmul(K_mn,b)
    b = 1/tf.matmul(tf.transpose(K_mn),a)

G_loss2 = tf.reduce_sum(  tf.matmul(K_mn*C_mn,b) * a)   # * element-wise

total_loss =  G_loss1 + G_loss2

train_variables = tf.trainable_variables()
generator_variables = [v for v in train_variables if v.name.startswith("Generator")]
G_solver = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(total_loss, var_list=generator_variables)


Z_fix = tf.constant(np.random.uniform(low=0.0, high=1.0, size=(3000,Z_dim)),dtype=data_type)
Fixed_sample = generator(Z_fix,'Generator',True)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

samples_circle,samples_square = sess.run([real_data_circle,real_data_square])
fig = plt.figure()
plt.scatter(samples_circle[:,0],samples_circle[:,1])
plt.savefig('out/{}_{}.png'
            .format('real_circle',config), bbox_inches='tight')
plt.close(fig)
fig = plt.figure()
plt.scatter(samples_square[:,0],samples_square[:,1])
plt.savefig('out/{}_{}.png'
            .format('real_square',config), bbox_inches='tight')
plt.close(fig)

for it in range(ITERATIONS):
    if it==0:
        epsilon_ = epsilon_list[0]
    if it==500:
        epsilon_ = epsilon_list[1]
    if it==1000:
        epsilon_ = epsilon_list[2]
    if it==2000:
        epsilon_ = epsilon_list[3]
    if it==3000:
        epsilon_ = epsilon_list[4]

  
    _,G_loss_curr,u,v,K_ = sess.run([ G_solver,total_loss,a,b,K_mn], feed_dict={ epsilon:epsilon_} )


    if it % 100 == 0:
        print('Iter: {};  G_loss: {:.4}; '.format(it, G_loss_curr))
        if it % 1000 == 0:
            samples = sess.run(Fixed_sample)
            fig = plt.figure()
            plt.scatter(samples[:,0],samples[:,1])
            plt.savefig('out/{}_{}_{}_{}.png'
                        .format(DATA,'sinkhorn_mixture_',config,str(it).zfill(3)), bbox_inches='tight')
            plt.close(fig)
  
    