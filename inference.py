import tensorflow as tf


IMAGE_SIZE = 96
NUM_CHANNELS = 1
NUM_LANDMARKS = 15

CONV1_DEEP = 32
CONV1_SIZE = 3

CONV2_DEEP = 64
CONV2_SIZE = 3

CONV3_DEEP = 128
CONV3_SIZE = 2

FC1_SIZE = 500
FC2_SIZE = 500


def get_weights(size):
    return tf.get_variable('weights', size, initializer=tf.truncated_normal_initializer(stddev=0.1))


def get_biases(size):
    return tf.get_variable('biases', size, initializer=tf.constant_initializer(0.0))


def inference(input_tensor, train):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = get_weights(
            [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP])
        conv1_biases = get_biases([CONV1_DEEP])

        conv1 = tf.nn.conv2d(input_tensor, conv1_weights,
                             strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.variable_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[
                               1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('layer3-conv2'):
        conv2_weights = get_weights(
            [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP])
        conv2_biases = get_biases([CONV2_DEEP])

        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[
                             1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.variable_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[
                               1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('layer5-conv3'):
        conv3_weights = get_weights(
            [CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP])
        conv3_biases = get_biases([CONV3_DEEP])

        conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[
                             1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))

    with tf.variable_scope('layer6-pool3'):
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[
                               1, 2, 2, 1], padding='SAME')

    pool_shape = pool3.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool3, [-1, nodes])

    with tf.variable_scope('layer7-fc1'):
        fc1_weights = get_weights([nodes, FC1_SIZE])
        fc1_biases = get_biases([FC1_SIZE])

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer8-fc2'):
        fc2_weights = get_weights([FC1_SIZE, FC2_SIZE])
        fc2_biases = get_biases([FC2_SIZE])

        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        if train:
            fc2 = tf.nn.dropout(fc2, 0.5)

    with tf.variable_scope('layer9-fc3'):
        fc3_weights = get_weights([FC2_SIZE, NUM_LANDMARKS * 2])
        fc3_biases = get_biases([NUM_LANDMARKS * 2])
        targets = tf.matmul(fc2, fc3_weights) + fc3_biases

    return targets
