from __future__ import print_function
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from pandas import *

learning_rate = 0.013
REG=10
num_steps = 100
batch_size = int(mnist.train.num_examples/5)
display_step = 1

# Network Parameters
n_hidden_1 = 1024 # 1st layer number of neurons
n_hidden_2 = 512 # 2nd layer number of neurons
num_features = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [num_features,None])
Y = tf.placeholder("float", [num_classes,None])

def weight_scale(Lin,Lout):
    scale=np.sqrt(6)/(np.sqrt(Lout)+np.sqrt(Lin))
    print(scale)
    return scale

# Store layers weight & bias
weights = {
    'W1': tf.Variable(tf.random_normal([n_hidden_1,num_features],stddev=weight_scale(n_hidden_1,num_features))),
    'W2': tf.Variable(tf.random_normal([n_hidden_2,n_hidden_1],stddev=weight_scale(n_hidden_2,n_hidden_1))),
    'W3': tf.Variable(tf.random_normal([num_classes,n_hidden_2],stddev=weight_scale(num_classes,n_hidden_2)))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1,1],stddev=weight_scale(n_hidden_1,0))),
    'b2': tf.Variable(tf.random_normal([n_hidden_2,1],stddev=weight_scale(n_hidden_2,0))),
    'b3': tf.Variable(tf.random_normal([num_classes,1],stddev=weight_scale(num_classes,0)))
}

# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(weights['W1'],x), biases['b1'])
    a_1=tf.nn.relu(layer_1)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(weights['W2'],a_1), biases['b2'])
    a_2=tf.nn.relu(layer_2)
    # Output fully connected layer with a neuron for each class
    layer_out = tf.add(tf.matmul(weights['W3'],a_2) , biases['b3'])
    a_out=tf.nn.softmax(layer_out)
    return a_out


# Construct model
prediction = neural_net(X)
print("size prediction, Y= ", prediction.shape, Y.shape)
# Define loss and optimizer
cost_fun = tf.reduce_mean(-tf.reduce_sum(
                        tf.math.log(prediction)*Y -(1-Y)*tf.math.log(1-prediction) 
                        ,axis=0))

#cost_fun= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=Y))
L2n= lambda x: tf.reduce_mean(x*x)
cost_fun=cost_fun+ REG* (L2n(weights['W3'])+L2n(weights['W2'])+L2n(weights['W1']))

decayed_learning_rate = tf.train.exponential_decay(learning_rate, num_steps,
                                                    0.7*num_steps, 0.96, staircase=True)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_fun)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 0), tf.argmax(Y, 0))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    
    for step in range(1, num_steps+1):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)

        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_y = sess.run(tf.one_hot(batch_y,10))

            # Run optimization op (backprop)
            _,c,acc=sess.run([optimizer,cost_fun,accuracy], feed_dict={X: batch_x.T, Y: batch_y.T})
            avg_cost += c / total_batch
        
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            print("Step " + str(step) + ", Loss= " + \
                  "{:.4f}".format(avg_cost) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")
    print ("mnist.test.images:", mnist.test.images.shape, "mnist.test.labels", mnist.test.labels.shape);

    x_test=mnist.test.images.T
    y_test=sess.run(tf.one_hot(mnist.test.labels,10))#the sess.run is to convert a tensor to np array
    # Calculate accuracy for MNIST test images
    acc_test,pre_test= sess.run(
                                [accuracy,prediction], 
                                feed_dict={X: mnist.test.images.T,Y: y_test.T}
                                )
    print("Testing Accuracy:", acc_test)

    print("sample label:", np.argmax(y_test.T[:,0]), "prediction label=", np.argmax(pre_test[:,0]))
