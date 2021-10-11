import tensorflow as tf
import numpy as np
import pdb
# for not mutually exclusive classes
def nll_loss_sigmoid(gt,logits,weights=[]):
    with tf.variable_scope('nll_sigmoid'):
        eps = 1e-9
        pred = tf.sigmoid(logits)
        gt = tf.cast(gt, tf.float32 )
        positive_block = gt * tf.log(eps+pred)
        negative_block = (1.0-gt)*tf.log(eps+1.0-pred)
        log_likelihood = -(positive_block + negative_block)
        nll_loss = tf.reduce_mean(log_likelihood)
    return nll_loss, pred
# for  mutually exclusive classes including background
def nll_loss_softmax(gt,logits,weights=[]):
    with tf.variable_scope('nll_softmax'):
        eps = 1e-9
        pred = tf.math.softmax(logits)
        gt = tf.cast(gt, tf.float32 )
        log_likelihood = - gt * tf.log(tf.clip_by_value(pred, eps,1e9))
        log_likelihood = tf.reduce_sum(log_likelihood,axis=len(gt.get_shape().as_list())-1)
        nll_loss = tf.reduce_mean(log_likelihood)
    return nll_loss, pred

# for not mutually exclusive classes, loss is weighted based on classes and automatically calculates the weights bckgr vs foreground
def weighted_nll_loss_sigmoid(gt,logits,weights=[]):
    with tf.variable_scope('wnll_sigmoid'):
        eps = 1e-9
        pred = tf.sigmoid(logits)
        axis = np.arange(0,len(pred.get_shape().as_list())-1).tolist()
        b_weights = tf.divide(tf.cast(tf.reduce_prod(gt.get_shape().as_list()),tf.float32),tf.cast(tf.reduce_sum(tf.cast(tf.equal(gt,1),tf.uint8),axis=axis),tf.float32))
        nll_loss = tf.add(tf.multiply(tf.multiply(tf.cast(gt, tf.float32),-tf.log(tf.add(pred,eps))), b_weights),tf.multiply(tf.subtract(1.0, tf.cast(gt, tf.float32)), -tf.log(tf.add(tf.subtract(1.0, pred),eps))))
        nll_loss = tf.multiply(weights,nll_loss)
        nll_loss = tf.reduce_sum(nll_loss,axis=len(gt.get_shape().as_list())-1)
        nll_loss = tf.reduce_mean(nll_loss)
    return nll_loss, pred

# for mutually exclusive classes loss is weighted
def weighted_nll_loss_softmax(gt,logits,weights=[]):
    with tf.variable_scope('wnll_softmax'):
        eps = 1e-9
        pred = tf.math.softmax(logits)
        log_likelihood = -tf.multiply( tf.cast(gt, tf.float32 ), tf.log( tf.add(pred,eps) ) )
        log_likelihood = tf.multiply(weights,log_likelihood)
        log_likelihood = tf.reduce_sum(log_likelihood,axis=len(gt.get_shape().as_list())-1)
        nll_loss = tf.reduce_mean(log_likelihood)
    return nll_loss, pred

# for not mutually exclusive classes
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

# for mutually exclusive classes
def dice_loss_softmax(gt,logits,weights=[]):
    with tf.variable_scope('dice_softmax'):
        eps = 1e-9
        gt = tf.cast( gt, tf.float32 )
        pred = tf.nn.softmax(logits)
        axis = np.arange(0,len(pred.get_shape().as_list())-1).tolist()
        top = gt * pred
        intersection = tf.reduce_sum( top , axis = axis)
        left = tf.reduce_sum( gt, axis = axis )
        right = tf.reduce_sum( pred, axis = axis )
        union = tf.add( left, right )
        dice_class =  2.0* intersection / (tf.clip_by_value(union, 1e-9,1e9))
        dice_loss = - tf.log(tf.reduce_mean(dice_class))
    return dice_loss, pred

# for not mutually exclusive classes
def weighted_dice_loss_sigmoid(gt,logits,weights=[]):
    with tf.variable_scope('wdice_sigmoid'):
        eps = 1e-9
        pred = tf.sigmoid(logits)
        axis = np.arange(0,len(pred.get_shape().as_list())-1).tolist()
        top = tf.multiply( tf.cast( gt, tf.float32 ), pred )
        intersection = tf.reduce_sum( top , axis = axis)
        left = tf.reduce_sum( tf.cast( gt, tf.float32 ), axis = axis )
        right = tf.reduce_sum( pred, axis = axis )
        union = tf.add( left, right )
        dice_class = tf.divide( tf.multiply( 2.0, intersection ), tf.add( union, eps ) )
        dice_class = tf.multiply(weights,dice_class)
        dice_loss = - tf.log(tf.reduce_mean(dice_class))
    return dice_loss, pred

# for mutually exclusive classes
def weighted_dice_loss_softmax(gt,logits,weights=[]):
    with tf.variable_scope('wdice_softmax'):
        eps = 1e-9
        pred = tf.math.softmax(logits)
        axis = np.arange(0,len(pred.get_shape().as_list())-1).tolist()
        top = tf.multiply( tf.cast( gt, tf.float32 ), pred )
        intersection = tf.reduce_sum( top , axis = axis)
        left = tf.reduce_sum( tf.cast( gt, tf.float32 ), axis = axis )
        right = tf.reduce_sum( pred, axis = axis )
        union = tf.add( left, right )
        dice_class = tf.divide( tf.multiply( 2.0, intersection ), tf.add( union, eps ) )
        dice_class = tf.multiply(weights,dice_class)
        dice_loss = - tf.log(tf.reduce_mean(dice_class))
    return dice_loss, pred

# for mutually exclusive classes
def wCE_dice(gt,logits,weights=[],lamb=0.1):
    with tf.variable_scope('wCE_dice'):
        wce_loss, pred = weighted_nll_loss_softmax(gt,logits,weights=weights)
        dice_loss,_ = dice_loss_softmax(gt,logits)
        combo_loss = tf.add(tf.multiply(wce_loss,lamb),dice_loss)
    return combo_loss, pred

# for mutually exclusive classes
def wCE_wdice(gt,logits,weights=[],lamb=0.1):
    with tf.variable_scope('wCE_wdice'):
        wce_loss, pred = weighted_nll_loss_softmax(gt,logits,weights=weights)
        dice_loss,_ = weighted_dice_loss_softmax(gt,logits,weights=weights)
        combo_loss = tf.add(tf.multiply(wce_loss,lamb),dice_loss)
    return combo_loss, pred

def wCE_wdice_sigmoid(gt,logits,weights=[],lamb=0.1):
    with tf.variable_scope('wCE_wdice'):
        wce_loss, pred = weighted_nll_loss_sigmoid(gt,logits,weights=weights)
        dice_loss,_ = weighted_dice_loss_sigmoid(gt,logits,weights=weights)
        combo_loss = tf.add(tf.multiply(wce_loss,lamb),dice_loss)
    return combo_loss, pred

def penalise_confident_output_loss(gt,logits, weights=[]):
    with tf.variable_scope('penalise_confident_output_loss'):
        pred      = tf.math.softmax(logits)
        model_out = tf.add(pred,1e-9)
        ceLoss = tf.multiply(tf.cast(gt,tf.float32),- tf.log(model_out))

        ceLoss = tf.reduce_sum(ceLoss,axis=len(gt.get_shape().as_list())-1)
        entropy_per_class = tf.multiply(model_out,-tf.log(model_out))
        entropy = tf.reduce_sum(entropy_per_class,axis=len(gt.get_shape().as_list())-1)

        penalise_loss = tf.reduce_mean(tf.add(ceLoss,entropy))
    return penalise_loss, pred

def penalise_confident_output_sigmoid_loss(gt,logits, weights=[]):
    with tf.variable_scope('penalise_confident_output_sigmoid_loss'):
        eps = 1e-9
        pred = tf.sigmoid(logits)
        model_out = tf.add(pred,eps)
        positive_block = tf.multiply( tf.cast(gt, tf.float32 ), tf.log( model_out ) )
        negative_block = tf.multiply( tf.subtract( 1.0, tf.cast(gt, tf.float32 ) ),tf.log( tf.add(tf.subtract( 1.0 , pred ),eps) ) )
        ceLoss = -tf.add( positive_block, negative_block )

        entropy = tf.multiply(model_out,-tf.log(model_out))

        penalise_loss = tf.reduce_mean(tf.add(ceLoss,entropy))
    return penalise_loss, pred

def ce_sigmoid_weighted_distance(gt, logits, map, weights = []):
    with tf.variable_scope('ce_sigmoid_weighted_distance'):
        eps = 1e-9
        pred = tf.sigmoid(logits)
        model_out = tf.add(pred,eps)
        positive_block = tf.multiply(weights,tf.multiply( tf.cast(gt, tf.float32 ), tf.log( model_out )))
        negative_block = tf.multiply( tf.subtract( 1.0, tf.cast(gt, tf.float32 ) ),tf.log( tf.add(tf.subtract( 1.0 , pred ),eps) ) )
        ce = -tf.add( positive_block, negative_block )

        ce_dist = tf.multiply(ce, map)

        ceLoss = tf.reduce_sum(ce_dist,axis=len(gt.get_shape().as_list())-1)

        loss = tf.reduce_mean(ceLoss)
    return loss, pred

def ce_softmax_weighted_distance_MIDL(gt, logits, map, weights = []):
    with tf.variable_scope('ce_softmax_weighted_distance'):
        eps = 1e-9
        gt = tf.cast(gt, tf.float32 )
        pred = tf.math.softmax(logits)
        model_out = tf.add(pred,eps)
        positive_block = - weights * gt * tf.log(model_out)

        ceLoss = tf.reduce_sum(positive_block,axis=len(gt.get_shape().as_list())-1)

        ce_dist = ceLoss * map[...,0]
        loss = tf.reduce_mean(ce_dist)
    return loss, pred

def ce_softmax_weighted_distance(gt, logits, map, weights = []):
    with tf.variable_scope('ce_softmax_weighted_distance'):
        eps = 1e-9
        gt = tf.cast(gt, tf.float32 )
        pred = tf.math.softmax(logits)
        model_out = tf.add(pred,eps)
        positive_block = - weights * gt * tf.log(model_out)

        ce_dist = positive_block * map

        ceLoss = tf.reduce_sum(ce_dist,axis=len(gt.get_shape().as_list())-1)

        loss = tf.reduce_mean(ceLoss)
    return loss, pred

# for mutually exclusive classes
def dice_loss_softmax_distance(gt, logits, map, weights=[]):
    with tf.variable_scope('dice_softmax_distance'):
        gt = tf.cast( gt, tf.float32 )
        pred = tf.math.softmax(logits)
        axis = np.arange(0,len(pred.get_shape().as_list())-1).tolist()
        top = tf.multiply( gt, pred )
        intersection = tf.reduce_sum( top , axis = axis)
        left = tf.reduce_sum( gt, axis = axis )
        right = tf.reduce_sum( pred, axis = axis )
        union = left + right
        dice_class = (2.0 * intersection )/(tf.clip_by_value(union, 1e-9,1e9))
        dice_class = dice_class * map
        dice_class = weights * dice_class
        dice_loss = - tf.log(tf.reduce_mean(dice_class))
    return dice_loss, pred

# for not mutually exclusive classes and binary
def dice_loss_sigmoid_distance(gt, logits, map, weights=[]):
    with tf.variable_scope('dice_sigmoid_distance'):
        gt = tf.cast( gt, tf.float32 )
        pred = tf.sigmoid(logits)
        axis = np.arange(0,len(pred.get_shape().as_list())-1).tolist()
        top = tf.multiply( gt, pred )
        intersection = tf.reduce_sum( top , axis = axis)
        left = tf.reduce_sum( gt, axis = axis )
        right = tf.reduce_sum( pred, axis = axis )
        union = left + right
        dice_class = (2.0 * intersection )/(tf.clip_by_value(union, 1e-9,1e9))
        dice_class = dice_class * map
        dice_class = weights * dice_class
        dice_loss = - tf.log(tf.reduce_mean(dice_class))
    return dice_loss, pred
