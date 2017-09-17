import tensorflow as tf
import datasets
import inference


TRAIN_FILE = 'kaggle/training.csv'
TEST_FILE = 'kaggle/test.csv'
SAVE_PATH = 'model'
SUMMARY_PATH = 'log'

VALIDATION_SIZE = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-03
MOVING_AVERAGE_DECAY = 0.99
TRAINING_STEPS = 2000


def train(data):
    with tf.name_scope('input'):
        x = tf.placeholder(
            tf.float32,
            [None, inference.IMAGE_SIZE, inference.IMAGE_SIZE, inference.NUM_CHANNELS],
            name='x-input'
        )
        y_ = tf.placeholder(
            tf.float32,
            [None, inference.NUM_LANDMARKS * 2],
            name='y-input'
        )
        op = tf.placeholder(tf.bool)

    y = inference.inference(x, op)
    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.name_scope('rmse'):
        rmse = tf.sqrt(tf.reduce_mean(tf.square(y - y_)))
        tf.summary.scalar('rmse', rmse)
    with tf.name_scope('r_squre'):
        total_error = tf.reduce_sum(tf.square(y - tf.reduce_mean(y)))
        unexplained_error = tf.reduce_sum(tf.square(y - y_))
        r_square = 1 - tf.divide(unexplained_error, total_error)
        tf.summary.scalar('r_squre', r_square)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(
            LEARNING_RATE).minimize(rmse, global_step=global_step)
        train_op = tf.group(train_step, variable_averages_op, name='train')

    merged = tf.summary.merge_all()
    # saver = tf.train.Saver()
    train_summary_writer = tf.summary.FileWriter(
        SUMMARY_PATH + '/train', tf.get_default_graph())
    valid_summary_writer = tf.summary.FileWriter(SUMMARY_PATH + '/valid')

    def feed_dict(train):
        if train:
            xs, ys = data['train'].next_batch(BATCH_SIZE)
            return {x: xs, y_: ys, op: True}
        else:
            images = data['valid'].images
            targets = data['valid'].targets
            return {x: images, y_: targets, op: False}

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            # _, rmse_value, r2_score, step = sess.run(
            #     [train_op, rmse, r_square, global_step], feed_dict={x: xs, y_: ys})
            summary, _ = sess.run([merged, train_op],
                                  feed_dict=feed_dict(train=True))
            train_summary_writer.add_summary(summary, i)

            if i % 10 == 0:
                summary, r2_score = sess.run(
                    [merged, r_square], feed_dict=feed_dict(train=False))
                valid_summary_writer.add_summary(summary, i)
        train_summary_writer.close()
        valid_summary_writer.close()

        # if i % 100 == 0:
        #     rmse_valid, r2_valid = sess.run([rmse, r_square], feed_dict={
        #                                     x: valid_data.images, y_: valid_data.targets})
        #     print('Step {:d}: rmse(train): {:.6f}, r_square(train): {:.6f}, rmse(valid):{:.6f},r_squared(valid):{:.6f}'.format(
        #         step, rmse_value, r2_score, rmse_valid, r2_valid))


if __name__ == '__main__':
    data = datasets.kaggle_data(TRAIN_FILE, 0.3)
    train(data)
