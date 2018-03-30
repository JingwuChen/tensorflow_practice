import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.data.Dataset.from_tensor_slices(filenames)
dataset = dataset.flat_map(
    lambda filename: (
        tf.data.TextLineDataset(filename)
        .skip(1)
        .filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), "#"))))
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
training_filenames = ["iris.csv"]
with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict={filenames: training_filenames})
    for _ in range(0,5):
        print(sess.run(next_element))