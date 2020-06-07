import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy
# from pandas import *
import pandas as pd

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

data_x = np.loadtxt("X.txt")
data_y = np.loadtxt("y.txt")%10

msk = np.random.rand(len(data_x)) < 0.8
train_x = data_x[msk]
train_y = data_y[msk]
test_x = data_x[~msk]
test_y = data_y[~msk]
total_train_set=train_x.shape[0]
m=total_train_set;
print('number of training set:',m)

learning_rate = 0.002
num_steps = 300
display_step = 50

# Network Parameters
n_hidden_1 = 25 # 1st layer number of neurons
num_input = 400 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])
lambda_reg = tf.constant(0.5, dtype=tf.float32)


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1],0,0.02)),
    'out': tf.Variable(tf.random_normal([n_hidden_1, num_classes],0,0.02))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1],0.0,0.02)),
    'out': tf.Variable(tf.random_normal([num_classes],0.0,0.02))
}

# Create model
def neural_net(x):
    # Hidden fully connected layer with 25 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1'],transpose_a=False,transpose_b=False), biases['b1'])
    a_2=tf.nn.sigmoid(layer_1)
    theta1_reg_term =tf.matmul(x, weights['h1'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    # a_3=tf.nn.sigmoid(out_layer)
    a_3=tf.nn.softmax(out_layer)
    theta2_reg_term =tf.matmul(layer_1, weights['out'])
    return a_3,theta1_reg_term,theta2_reg_term


# Construct model
a_out,theta1_reg_term,theta2_reg_term = neural_net(X)

## Note that the original cost function introduced in the lectures is kind of not well behaved and needs
## some advanced optimization technique, for this reason we are not going to use that
loss_op = -tf.reduce_sum(tf.reduce_sum(Y*tf.log(a_out))\
            + (1-Y)*tf.log(1-a_out))/m \
            +tf.reduce_sum(tf.reduce_sum(theta1_reg_term*theta1_reg_term))*lambda_reg/(2.0*m) \
            +tf.reduce_sum(tf.reduce_sum(theta2_reg_term*theta2_reg_term))*lambda_reg/(2.0*m)

## Instead, we are using the softmax_cross_entropy_with_logits
# loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=a_out, labels=Y))



# Define loss and optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(a_out, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    # optimizer.minimize(sess)

    for step in range(1, num_steps+1):
        batch_x = train_x
        batch_y = sess.run(tf.one_hot(train_y , 10 ))
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            # print (batch_x.shape, batch_y.shape, test.shape)
            print("Step " + str(step) + ", Cost= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Training Finished!")
    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: test_x,
                                      Y: sess.run(tf.one_hot(test_y , 10))}))

    from random import *
    while True:
        n = input("\nDo you want to try more (Y/n):")
        if n.strip() == 'n':
            break
        else:
            r=randint(1, len(test_x))
            x=np.reshape(test_x[r,:],(20,20))
            from matplotlib import pyplot as plt
            plt.imshow(x, interpolation='nearest')
            plt.show()
            pred=sess.run(tf.argmax(a_out,1),feed_dict={X: test_x[r,:].reshape(1,400)})
            print ('NN prediction for this image: ', pred[0])
