import tensorflow as tf
import numpy as np
import input_data
import sys

sys.dont_write_bytecode = True

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
OUTPUT_SIZE = 10
LAYERS = 50

# 2D conv
def conv2d (x, W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

# initialize a weight variable
def createWeight(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

# initialize a bias variable
def createBias(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def max_pool_2x2 (x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

# implementing the suggested network layer
def DanCNN (input_channel_num, input_image):
    W_conv_1x1 = createWeight ([1,1,input_channel_num,1])
    b_conv_1x1 = createBias ([1])

    h_conv_1x1 = tf.nn.relu(conv2d(input_image, W_conv_1x1) + b_conv_1x1)

    W_conv_3x3 = createWeight ([3,3,1,1])
    b_conv_3x3 = createBias ([1])

    h_conv_3x3 = tf.nn.relu(conv2d(h_conv_1x1,W_conv_3x3) + b_conv_3x3)

    output_channel_num = input_channel_num + 1

    # output = max_pool_2x2 (tf.concat(3,[input_image, h_conv_3x3]))
    output = tf.concat (3, [input_image, h_conv_3x3])
    return output


def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.scalar_summary('stddev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)

def main():
    mnist = input_data.read_data_sets("MNIST.data/", one_hot = True)
    # create model
    x = tf.placeholder(tf.float32, [None, IMAGE_SIZE])
    y = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])
    sess = tf.InteractiveSession()
    x_image = tf.reshape(x,[-1,28,28,1])
    x_temp = x_image
    for i in range (1, LAYERS):
        x_temp = DanCNN (i, x_temp)
        if (i == 10 or i == 30 or i == 40):
            x_temp = max_pool_2x2 (x_temp)
        # print x_temp
    W_fc1 = createWeight ([4*4, 16])
    b_fc1 = createBias ([16])

    print x_temp
    x_temp = tf.reduce_mean(x_temp,3)
    print x_temp
    x_temp_flat = tf.reshape (x_temp, [-1, 4*4])
    h_fc1 = tf.nn.relu(tf.matmul(x_temp_flat, W_fc1) + b_fc1)

    W_fc2 = createWeight ([16,10])
    b_fc2 = createBias ([10])

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    y_conv = tf.nn.softmax(h_fc2)

# train and evaluation
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_conv), reduction_indices = 1))
        tf.scalar_summary('cross entropy', cross_entropy)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))

    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary('accuracy', accuracy)

    merged = tf.merge_all_summaries()

    train_writer = tf.train.SummaryWriter('log',sess.graph)
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())

    for i in range(100000):
        batch = mnist.train.next_batch(50)
        if i%100== 0:
	    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y: batch[1]})
	    train_cross_entropy = cross_entropy.eval(feed_dict={x:batch[0], y: batch[1]})
	    with open('log/data.txt',"a") as output_file:
		output_file.write("{},{} {}\n".format(i,train_accuracy, train_cross_entropy))
        if i%10000== 0 and i != 0:
            saver.save(sess, "tmp/model.ckpt")
        print("step %d, training accuracy %g"%(i, train_accuracy))
        summary,_ = sess.run([merged, train_step],feed_dict={x: batch[0], y: batch[1]})
        train_writer.add_summary(summary,i)
    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: mnist.test.images, y: mnist.test.labels}))

if __name__ == '__main__':
    main()
