import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pandas import *

rng = np.random

# Parameters
learning_rate = 0.001
training_epochs = 200000
display_step = 5000

# Training Data
data = np.loadtxt("ex2data1.txt", delimiter=",")


train_X=data[:,0:2]
m=train_X.shape[0]
train_Y=data[:,2].reshape(m,1)

# tf Graph Input
X = tf.placeholder(tf.float32, [None, 2]) #num features,
Y = tf.placeholder(tf.float32, [None, 1]) #num labels 

# Set model weights
W = tf.Variable(tf.zeros([2, 1]), name='weights') #num features, num labels 
b = tf.Variable(0.0, name='biases')

# Construct a  model
pred = tf.nn.sigmoid(tf.matmul(X, W)+ b)
linear = tf.matmul(X, W)+ b

# Minimize error using cross entropy
cost = -tf.reduce_sum(Y*tf.log(pred)+ (1-Y)*tf.log(1-pred) ,[0])/m
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


    # Only need 2 points to define a line, so choose two endpoints
    plot_x = np.array([np.min(train_X[:,1])-2,  np.max(train_X[:,1])+2])
    # Calculate the decision boundary line
    theta=sess.run(W)
    plot_y = -1.0/theta[1]*(theta[0]*plot_x + sess.run(b) ) 

    #Graphic display
    plt.plot(train_X[:,0], train_X[:,1], 'ro', label='Original data')
    plt.plot(plot_x,plot_y ,label='Decision boundary')
    plt.legend()
    plt.show()