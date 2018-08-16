import itertools
import tensorflow as tf
tf.enable_eager_execution()


def gen():
  for i in itertools.count(1):
    yield (i, [1] * i)


ds = tf.data.Dataset.from_generator(gen, (tf.int64, tf.int64), (tf.TensorShape([]), tf.TensorShape([None])))
print(ds.make_one_shot_iterator())
# value = ds.make_one_shot_iterator().get_next()

# with tf.Session() as sess:
#   print(sess.run(value))  # (1, array([1]))
#   print(sess.run(value))  # (2, array([1, 1]))
# # print(value)
