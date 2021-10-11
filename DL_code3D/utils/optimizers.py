import tensorflow as tf

def adam(lr,global_step):
    opt = tf.train.AdamOptimizer( learning_rate = lr )
    return opt


def sgd(lr,global_step,decay_steps=2000,decay_rate=0.7):
    lr = tf.train.exponential_decay(lr, global_step, decay_steps, decay_rate,
        staircase=True)
    opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
    return opt

def momentum_optimizer(lr, global_step, decay_steps=1000, lr_decay_rate=0.8, momentum_decay_rate=0.9):
    lr = tf.train.exponential_decay(lr, global_step, decay_steps, lr_decay_rate, staircase=True)
    opt = tf.train.MomentumOptimizer(lr, momentum_decay_rate)
    return opt
