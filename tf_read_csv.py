import tensorflow as tf
# filename_queue=tf.train.string_input_producer(
#     tf.train.match_filenames_once("./*.csv"),shuffle=True)
filename_queue=tf.train.string_input_producer(["iris.csv",])
reader=tf.TextLineReader()
# reader=tf.TextLineReader(skip_header_lines=1)
key ,value=reader.read(filename_queue)
record_default=[[0.],[0.],[0.],[0.],[""]]
col1,col2,col3,col4,col5=tf.decode_csv(value,record_defaults=record_default)
features=tf.stack([col1,col2,col3,col4])
with tf.Session() as sess:
    # sess.run(tf.initialize_all_variables())
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(coord=coord,sess=sess)
    for i in range(12):
    # Retrieve a single instance:
        example, label = sess.run([features, col5])
        print(example,label)
    coord.request_stop()
    coord.join(threads)