import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pandas import *

rng = np.random

# Parameters
learning_rate = 0.001
training_epochs = 100000
display_step = 10000

# Training Data
data = np.loadtxt("ex2data2.txt", delimiter=",")

train_X=np.array(data[:,0:2])
m=train_X.shape[0]
train_Y=data[:,2].reshape(m,1)
degree = 6;
for i in range(1, degree+1):
    for j in range(0,i+1):
        size=train_X.shape[1]
        new_feature=np.power(data[:,0],(i-j)) *np.power(data[:,1],j)
        train_X=np.c_[train_X,new_feature]

# the first two column were created again so let's remove them
train_X=train_X[:,2:]       
numFeature=train_X.shape[1]

# tf Graph Input
X = tf.placeholder(tf.float32, [None, numFeature]) #num features,
Y = tf.placeholder(tf.float32, [None, 1]) #num labels 

# Set model weights
W = tf.Variable(tf.zeros([numFeature, 1]), name='weights') #num features, num labels 
b = tf.Variable(0.0, name='biases')
lambda_reg = tf.constant(1.0, dtype=tf.float32)

# Construct a  model
pred = tf.nn.sigmoid(tf.matmul(X, W)+ b)
linear = tf.matmul(X, W)+ b

# Minimize error using cross entropy + regularized term
cost = -tf.reduce_sum(Y*tf.log(pred)+ (1-Y)*tf.log(1-pred) ,[0])/m +lambda_reg*tf.reduce_sum(tf.multiply(W,W))/(2*m)
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    sess.run(init)
    # Fit all training data
    mcost=sess.run(cost,feed_dict={X: train_X, Y:train_Y})
    mW=sess.run(W)
    print '\n" Initial cost is :',mcost, '\nInitial W:\n',DataFrame(mW), '\nInitial b:\n',sess.run(b)

    for epoch in range(training_epochs):        
        _,c=sess.run([optimizer,cost],feed_dict={X: train_X, Y:train_Y})

        #Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print "Iteration:", '%04d' % (epoch+1), ", cost=",c[0]


    print "Optimization Finished!"
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print "Training cost=", training_cost[0], "\nW=\n", DataFrame(sess.run(W)), "\nb=\n", sess.run(b), '\n'

    #Graphic display
    plt.plot(train_X[:,0], train_X[:,1], 'ro', label='Original data')
    plt.xlabel('microchip test 1')
    plt.ylabel('microchip test 1')
    plt.title(r'$\lambda=$%1.1f'%sess.run(lambda_reg))
    plt.legend()
    plt.show()