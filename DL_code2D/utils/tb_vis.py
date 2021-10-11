
import tensorflow as tf
import pdb
def summarize3D(comp,key_slice,type,input,gt,pred,r,c,loss,dice_score,dice_classes):
    cat = []
    for s in key_slice:
        cat.append(tf.reshape(input[:,:,:,s,0],[-1,r,c,1]))
    mri = tf.concat(cat,2)

    tb_summary = {}
    tb_summary['mri'] = tf.summary.image(type+'/MRI', mri)
    tb_summary['loss'] = tf.summary.scalar(type+'/loss', loss)
    tb_summary['dice_acc'] = tf.summary.scalar(type+'/dice_acc', dice_score)

    for l,x in enumerate(comp):
        r_gt = tf.multiply(tf.cast(tf.reshape(gt[:,:,:,key_slice[l],l],[-1,r,c,1]),tf.uint8),tf.constant(255,dtype = tf.uint8))
        g_pred = tf.multiply(tf.cast(tf.reshape(pred[:,:,:,key_slice[l],l],[-1,r,c,1]),tf.uint8),tf.constant(255,dtype = tf.uint8))
        b = tf.multiply(tf.ones(r_gt.get_shape(), dtype = tf.uint8),tf.constant(255,dtype = tf.uint8))
        a = tf.cast(tf.maximum(tf.cast(r_gt,tf.float32),tf.cast(g_pred,tf.float32)),tf.uint8)
        rgb = tf.concat([r_gt,g_pred,b,a],3)
        tb_summary[x] = tf.summary.image(type+'/'+str(l)+'_'+x, rgb)
        tb_summary['dice_'+x] = tf.summary.scalar(type+'/dice_acc_'+x, dice_classes[l])

    summary_op = tf.summary.merge(list(tb_summary.values()))
    return summary_op

def summarize2D(comp,key_slice,type,input,gt,pred,r,c,loss,dice_score,dice_classes, lr):
    cat = tf.reshape(input,[-1,r,c,1])
    mri = tf.concat(cat,2)
    tb_summary = {}
    tb_summary['mri'] = tf.summary.image(type+'/MRI', mri)
    tb_summary['loss'] = tf.summary.scalar(type+'/loss', loss)
    tb_summary['dice_acc'] = tf.summary.scalar(type+'/dice_acc', dice_score)
    tb_summary['lr'] = tf.summary.scalar(type+'/lr', lr)
    for l,x in enumerate(comp):
        r_gt = tf.multiply(tf.cast(tf.reshape(gt[0,:,:,l],[-1,r,c,1]),tf.uint8),tf.constant(255,dtype = tf.uint8))
        g_pred = tf.multiply(tf.cast(tf.reshape(pred[0,:,:,l],[-1,r,c,1]),tf.uint8),tf.constant(255,dtype = tf.uint8))
        b = tf.multiply(tf.ones(r_gt.get_shape(), dtype = tf.uint8),tf.constant(255,dtype = tf.uint8))
        a = tf.cast(tf.maximum(tf.cast(r_gt,tf.float32),tf.cast(g_pred,tf.float32)),tf.uint8)
        rgb = tf.concat([r_gt,g_pred,b,a],3)
        tb_summary[x] = tf.summary.image(type+'/'+str(l)+'_'+x, rgb)
        tb_summary['dice_'+x] = tf.summary.scalar(type+'/dice_acc_'+x, dice_classes[l])

    summary_op = tf.summary.merge(list(tb_summary.values()))
    return summary_op
