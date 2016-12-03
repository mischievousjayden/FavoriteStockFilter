
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np

import datalayer as dl

import pdb

# read data
print("read data")
data_path = "../data"
data = dl.stockdata(data_path)

# print("length " + str(len(data.input_data)))
# print("length " + str(len(data.input_data["like"])))
# print("length " + str(len(data.input_data["dislike"])))
#
# pdb.set_trace()

# Parameters
n_cross_validation = 4
learning_rate = 0.001
training_iters = 1000
display_step = 10

# Network Parameters
n_input = 5 # the number of features
n_steps = 200 # the number of days
n_hidden = 128 # 128 # hidden layer num of features
n_classes = 2 # like or dislike

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# define weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

def cutData(data, num_row):
    features = []
    label = []
    for d in data:
        # features.append(np.array(d["features"])[:n_steps,:])
        features.append(np.flipud(np.flipud(d["features"])[:num_row,:]))
        label.append(d["label"])
    return [np.array(features), np.array(label)]

def createTrainTestData(cross_validation_data):
    train_features, train_label = cutData(cross_validation_data["train"], n_steps)
    test_features, test_label = cutData(cross_validation_data["test"], n_steps)
    return [train_features, train_label, test_features, test_label]


print("create lstm nn")
pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    for i in range(n_cross_validation):
        print("build train and test data")
        train_data, train_label, test_data, test_label = createTrainTestData(data.getCrossValidationInput(n_cross_validation, i))

        print("start learning")
        sess.run(init)
        step = 0
        display_flag = display_step
        # Keep training until reach max iterations
        while step < training_iters:
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: train_data, y: train_label})
            if display_flag < 0:
            # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: train_data, y: train_label})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: train_data, y: train_label})
                print("Iter " + str(step) + \
                      ", Minibatch Loss: " + "{:.6f}".format(loss) + \
                      ", Training Accuracy: " + "{:.5f}".format(acc))
                display_flag = display_step
            step += len(train_data)
            display_flag -= len(train_data)
        print("Optimization Finished!")

        # Calculate accuracy for 128 mnist test images
        print("Testing Accuracy:", \
                sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

