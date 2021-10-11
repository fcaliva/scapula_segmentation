import tensorflow as tf
def adam(lr,global_step):
    opt = tf.train.AdamOptimizer( learning_rate = lr )
    return opt
