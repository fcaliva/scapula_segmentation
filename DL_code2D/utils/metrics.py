import tensorflow as tf
import numpy as np
def spatial_dice(gt,pred,thresh=0):
    with tf.variable_scope('spatial_dice'):
        eps = 1e-9
        if thresh == 0:
            prediction = pred
        else:
            prediction = tf.cast(tf.greater(pred,thresh),tf.float32)
        axis = np.arange(0,len(pred.get_shape().as_list())-1).tolist()
        ctt = tf.multiply(tf.cast(gt, tf.float32),prediction)
        ctf = tf.multiply(tf.cast(gt, tf.float32), tf.subtract(1.0, prediction))
        cft = tf.multiply(tf.subtract(1.0, tf.cast(gt, tf.float32)), prediction)

        up = tf.reduce_sum(tf.add(ctf,cft), axis= axis)
        down1 = tf.reduce_sum(tf.multiply(2.0,ctt), axis= axis)
        down2 = tf.reduce_sum(cft, axis= axis)
        down3 = tf.reduce_sum(ctf, axis= axis)
        down = tf.add(down1, tf.add(down2, down3))

        dice_classes = tf.subtract(1.0, tf.divide(tf.add(up,eps), tf.add(down,eps)))
        dice_score = tf.reduce_mean(dice_classes)
    return dice_classes, dice_score

def dice_score(gt,pred,thresh=0):
    with tf.variable_scope('dice_score'):
        if thresh == 0:
            prediction = pred
        else:
            prediction = tf.cast(tf.greater(pred,thresh),tf.float32)
        axis = np.arange(0,len(pred.get_shape().as_list())-1).tolist()
        top = tf.cast(gt, tf.float32)*prediction
        intersection = tf.reduce_sum(top , axis = axis)
        left = tf.reduce_sum(tf.cast(gt, tf.float32), axis = axis)
        right = tf.reduce_sum(prediction, axis = axis)
        union = left+right
        eps = 1e-9
        dice_classes =  2.0* (tf.clip_by_value(intersection, 1e-9,1e9)) / (tf.clip_by_value(union, 1e-9,1e9))
        dice_score = tf.reduce_mean(dice_classes)
    return dice_classes, dice_score
