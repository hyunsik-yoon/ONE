import tensorflow as tf

in0_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 2), name="Hole0")
in1_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(1, 2), name="Hole1")
in2_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 2), name="Hole2")
add_ = tf.compat.v1.add(in0_, in1_, name="Add")
sub_ = tf.compat.v1.subtract(add_, in2_, name="Sub")
