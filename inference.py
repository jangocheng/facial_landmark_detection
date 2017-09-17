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
    return tf.Variable(tf.truncated_normal(size, stddev=0.1))


def get_biases(size):
    return tf.Variable(tf.constant(0.0, shape=size))


def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        tf.summary.histogram(name, var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.sqrt(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)


def conv_layer(input_tensor, input_channels, filter_length, filter_width, filter_depth, layer_name):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = get_weights(
                [filter_length, filter_width, input_channels, filter_depth])
            variable_summaries(weights, layer_name + '/weights')
        with tf.name_scope('biases'):
            biases = get_biases([filter_depth])
            variable_summaries(biases, layer_name + '/biases')
        with tf.name_scope('convolution'):
            conv = tf.nn.conv2d(input_tensor, weights, strides=[
                                1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, biases))
            tf.summary.histogram(layer_name + '/convolution', relu)
        return relu


def max_pool_layer(input_tensor, layer_name):
    with tf.name_scope(layer_name):
        with tf.name_scope('max_pool'):
            pool = tf.nn.max_pool(input_tensor, ksize=[1, 2, 2, 1], strides=[
                                  1, 2, 2, 1], padding='SAME')
            tf.summary.histogram(layer_name + '/max_pool', pool)
        return pool


def fc_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = get_weights([input_dim, output_dim])
            variable_summaries(weights, layer_name + '/weights')
        with tf.name_scope('biases'):
            biases = get_biases([output_dim])
            variable_summaries(biases, layer_name + '/biases')
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram(layer_name + '/pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram(layer_name + '/activations', activations)
        return activations


def inference(input_tensor, train):
    conv1 = conv_layer(input_tensor, NUM_CHANNELS, CONV1_SIZE,
                       CONV1_SIZE, CONV1_DEEP, 'layer1-conv1')
    pool1 = max_pool_layer(conv1, 'layer2-pool1')

    conv2 = conv_layer(pool1, CONV1_DEEP, CONV2_SIZE,
                       CONV2_SIZE, CONV2_DEEP, 'layer3-conv2')
    pool2 = max_pool_layer(conv2, 'layer4-pool2')

    conv3 = conv_layer(pool2, CONV2_DEEP, CONV3_SIZE,
                       CONV3_SIZE, CONV3_DEEP, 'layer5-conv3')
    pool3 = max_pool_layer(conv3, 'layer6-pool3')

    pool_shape = pool3.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool3, [-1, nodes])

    fc1 = fc_layer(reshaped, nodes, FC1_SIZE, 'layer7-fc1')
    # if train:
    #     fc1 = tf.nn.dropout(fc1, 0.5)
    fc1 = tf.cond(train, lambda: tf.nn.dropout(fc1, 0.5), lambda: fc1)

    fc2 = fc_layer(fc1, FC1_SIZE, FC2_SIZE, 'layer8-fc2')
    # if train:
    #     fc2 = tf.nn.dropout(fc2, 0.5)
    fc2 = tf.cond(train, lambda: tf.nn.dropout(fc2, 0.5), lambda: fc2)

    fc3 = fc_layer(fc2, FC2_SIZE, NUM_LANDMARKS *
                   2, 'layer9-fc3', act=tf.identity)

    return fc3

    # with tf.variable_scope('layer1-conv1'):
    #     conv1_weights = get_weights(
    #         [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP])
    #     conv1_biases = get_biases([CONV1_DEEP])

    #     conv1 = tf.nn.conv2d(input_tensor, conv1_weights,
    #                          strides=[1, 1, 1, 1], padding='SAME')
    #     relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # with tf.variable_scope('layer2-pool1'):
    #     pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[
    #                            1, 2, 2, 1], padding='SAME')

    # with tf.variable_scope('layer3-conv2'):
    #     conv2_weights = get_weights(
    #         [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP])
    #     conv2_biases = get_biases([CONV2_DEEP])

    #     conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[
    #                          1, 1, 1, 1], padding='SAME')
    #     relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # with tf.variable_scope('layer4-pool2'):
    #     pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[
    #                            1, 2, 2, 1], padding='SAME')

    # with tf.variable_scope('layer5-conv3'):
    #     conv3_weights = get_weights(
    #         [CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP])
    #     conv3_biases = get_biases([CONV3_DEEP])

    #     conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[
    #                          1, 1, 1, 1], padding='SAME')
    #     relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))

    # with tf.variable_scope('layer6-pool3'):
    #     pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[
    #                            1, 2, 2, 1], padding='SAME')

    # pool_shape = pool3.get_shape().as_list()
    # nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    # reshaped = tf.reshape(pool3, [-1, nodes])

    # with tf.variable_scope('layer7-fc1'):
    #     fc1_weights = get_weights([nodes, FC1_SIZE])
    #     fc1_biases = get_biases([FC1_SIZE])

    #     fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
    #     if train:
    #         fc1 = tf.nn.dropout(fc1, 0.5)

    # with tf.variable_scope('layer8-fc2'):
    #     fc2_weights = get_weights([FC1_SIZE, FC2_SIZE])
    #     fc2_biases = get_biases([FC2_SIZE])

    #     fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
    #     if train:
    #         fc2 = tf.nn.dropout(fc2, 0.5)

    # with tf.variable_scope('layer9-fc3'):
    #     fc3_weights = get_weights([FC2_SIZE, NUM_LANDMARKS * 2])
    #     fc3_biases = get_biases([NUM_LANDMARKS * 2])
    #     targets = tf.matmul(fc2, fc3_weights) + fc3_biases

    # return targets
