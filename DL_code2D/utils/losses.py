import tensorflow as tf
import numpy as np
def dice_loss_softmax(gt,logits,weights=[]):
    with tf.variable_scope('dice_softmax'):
        eps = 1e-9
        gt = tf.cast( gt, tf.float32 )
        pred = tf.math.softmax(logits)
        axis = np.arange(0,len(pred.get_shape().as_list())-1).tolist()
        top = gt * pred
        intersection = tf.reduce_sum( top , axis = axis)
        left = tf.reduce_sum( gt, axis = axis )
        right = tf.reduce_sum( pred, axis = axis )
        union = tf.add( left, right )
        dice_class =  2.0* (tf.clip_by_value(intersection, 1e-9,1e9)) / (tf.clip_by_value(union, 1e-9,1e9))
        dice_loss = - tf.log(tf.reduce_mean(dice_class))
    return dice_loss, pred

def nll_loss_sigmoid(gt,logits,weights=[]):
    with tf.variable_scope('nll_sigmoid'):
        eps = 1e-9
        pred = tf.sigmoid(logits)
        gt = tf.cast(gt, tf.float32 )
        positive_block = gt * tf.log(eps+pred)
        negative_block = (1.0-gt)*tf.log(eps+1.0-pred)
        log_likelihood = -(positive_block*.25 + negative_block)
        nll_loss = 0.1*tf.reduce_mean(log_likelihood)
    return nll_loss, pred

def dice_loss_sigmoid(gt,logits,weights=[]):
    with tf.variable_scope('dice_sigmoid'):
        eps = 1e-9
        pred = tf.sigmoid(logits)
        axis = np.arange(0,len(pred.get_shape().as_list())-1).tolist()
        top = tf.multiply( tf.cast( gt, tf.float32 ), pred )
        intersection = tf.reduce_sum( top , axis = axis)
        left = tf.reduce_sum( tf.cast( gt, tf.float32 ), axis = axis )
        right = tf.reduce_sum( pred, axis = axis )
        union = tf.add( left, right )
        dice_class = tf.divide( tf.add(tf.multiply( 2.0, intersection ), eps), tf.add( union, eps ) )
        dice_loss = - tf.log(tf.reduce_mean(dice_class))
    return dice_loss, pred
