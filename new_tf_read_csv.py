import tensorflow as tf
import os
#avoiding the annoying cpu warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#create a placeholder for the file name
filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.data.Dataset.from_tensor_slices(filenames)
# * Skip the first line (header row).
# * Filter out lines beginning with "#" (comments).
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