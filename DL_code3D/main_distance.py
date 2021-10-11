# DioscoriDESS was Greek Physician who worked in Rome in the 1st Century A.D.
# author of De Materia Medica. He recommended to use ivy for treatment of OA.
# main codebase written by Claudia and Francesco, slack us with any questions
import numpy as np
import os
import sys
sys.path.append('./utils')
import pdb  #pdb.set_trace()
import argparse
import yaml
import time
import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from logger import logger
from pprint import pprint

from utils import losses, early_stop, metrics, tb_vis, optimizers, tracker
import matplotlib.pyplot as plt
trial = False

if not trial:
    parser = argparse.ArgumentParser(description='define configuration file and run description')
    parser.add_argument('--cfg')
    parser.add_argument('--desc')
    args = parser.parse_args()
    with open(args.cfg) as f:
         config = yaml.load(f, Loader=yaml.UnsafeLoader)
    desc = args.desc
else:
    yaml_path = 'cfgs/training/MIDL_extra/experiment2.yaml'
    desc = 'distance_on_the_fly_test'
    with open(yaml_path) as f:
        config = yaml.load(f)


try:
    import importlib
    dataLoader = importlib.import_module(config['dataLoader_folder'])
except:
    print('using except dataloader, please update yaml config file')
    import dataLoader_main as dataLoader

try:
    sys.path.append('./models/'+config['model_folder'])
except:
    sys.path.append('./models/'+config['model'])
import network as nn

if not os.path.exists(config['common']['log_path']):
    os.makedirs(config['common']['log_path'])
if not os.path.exists(config['common']['save_path']):
    os.makedirs(config['common']['save_path'])

sys.stdout = logger(sys.stdout,path=config['common']['log_path'],desc=desc)
print('\n\n',sys.stdout.name,'\n\n')
pprint(config)
if 'all' not in config['common']['vis_GPU']:
    os.environ['CUDA_VISIBLE_DEVICES'] = config['common']['vis_GPU']
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
c = tf.ConfigProto()
#c.gpu_options.visible_device_list = "0"#GPU_VIS
c.gpu_options.allow_growth=True
c.gpu_options.per_process_gpu_memory_fraction = 0.95
c.allow_soft_placement = True
c.log_device_placement = False

seed = config['common']['seed']
np.random.seed(seed)
tf.reset_default_graph()
#tf.set_random_seed(seed)
model = nn.__dict__[config['model']](**config['model_params'])
global_step = tf.Variable(0, dtype=tf.int64, trainable=False)

input_shape = np.concatenate(([config['data_train']['batch_size']],config['data_train']['im_dims'],[config['data_train']['num_channels']]))
distance_shape = np.concatenate(([config['data_train']['batch_size']],config['data_train']['im_dims'],[config['data_train']['num_channels']]))
gt_shape = np.concatenate(([config['data_train']['batch_size']],config['data_train']['im_dims'],[config['data_train']['num_classes']]))

input = tf.placeholder(dtype=tf.float32, shape=input_shape)
gt = tf.placeholder(dtype=tf.uint8, shape=gt_shape)
distance = tf.placeholder(dtype=tf.float32, shape=distance_shape)
keep_prob = tf.placeholder(dtype=tf.float32, shape=())

loader_train = dataLoader.__dict__[config['learn']['dataloader']](**config['data_train'])
loader_val = dataLoader.__dict__[config['learn']['dataloader']](**config['data_val'])

logits = model.network_fn(input,keep_prob)
weights = tf.constant([config['learn']['weights']], dtype='float32')

optimizer = optimizers.__dict__[config['learn']['optimizer']](config['learn']['lr'],global_step)
if 'dist' in config['learn']['loss']:
    loss, pred = losses.__dict__[config['learn']['loss']](gt,logits,distance,weights)
else:
    loss, pred = losses.__dict__[config['learn']['loss']](gt,logits,weights)

score_classes, avg_score = metrics.__dict__[config['learn']['metrics']](gt,pred)

trainer = optimizer.minimize(loss = loss, global_step = global_step)

if len(config['data_train']['im_dims']) == 3:
    train_summary_op = tb_vis.summarize3D(config['learn']['comp'],config['learn']['key_slice'],'train', input, gt, pred, config['data_train']['im_dims'][0],config['data_train']['im_dims'][1],loss,avg_score,score_classes)
    val_summary_op = tb_vis.summarize3D(config['learn']['comp'],config['learn']['key_slice'],'val',input, gt, pred, config['data_train']['im_dims'][0],config['data_train']['im_dims'][1],loss,avg_score,score_classes)
elif len(config['data_train']['im_dims']) == 2:
    train_summary_op = tb_vis.summarize2D(config['learn']['comp'],config['learn']['key_slice'],'train', input, gt, pred, config['data_train']['im_dims'][0],config['data_train']['im_dims'][1],loss,avg_score,score_classes)
    val_summary_op = tb_vis.summarize2D(config['learn']['comp'],config['learn']['key_slice'],'val',input, gt, pred, config['data_train']['im_dims'][0],config['data_train']['im_dims'][1],loss,avg_score,score_classes)

init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

saver = tf.train.Saver(max_to_keep=15)

patience = early_stop.early_stop(patience=config['learn']['patience'])
with tf.Session(config=c) as sess:
    writer = tf.summary.FileWriter(config['common']['save_path']+sys.stdout.name,sess.graph)
    init_op.run()
    if config['pretrain']['flag']:
        saver.restore(sess, config['pretrain']['ckpt'])
        print('Restoring session')
        val_track = tracker.tracker(num_classes=config['data_val']['num_classes'])
        for viter in range(np.floor(loader_val.__len__()/config['data_val']['batch_size']).astype(int)):
            try:
                [ im_val, seg_val, dist_val, name_val ] = loader_val.fetch_batch()
            except:
                print('fetch_batch on restoring session did not work')
                continue
            val_summary, val_pred, val_loss, val_classes_score, val_score = sess.run([val_summary_op, pred, loss, score_classes, avg_score], feed_dict = {input: im_val, gt: seg_val, distance: dist_val, keep_prob:1.0})
            val_track.increment(val_loss,val_score,val_classes_score)
        val_cum_loss,val_cum_score,val_classes_cum_score = val_track.average()
        val_cum_loss_std,val_cum_score_std,val_classes_cum_score_std = val_track.stdev()
        _, _ = patience.track(val_cum_score)
        print('Restored successfully')

    print('Training...')
    train_track = tracker.tracker(num_classes=config['data_train']['num_classes'])
    for iter in range(config['learn']['max_steps']):
        try:
            [ im_train, seg_train, dist_train, name_train ] = loader_train.fetch_batch()
        except:
            print('fetch_batch on train set did not work')
            continue

        _, summary, current_loss, model_prediction, train_classes_score, train_score = sess.run([trainer, train_summary_op, loss, pred, score_classes, avg_score], feed_dict = {input: im_train, gt: seg_train, distance: dist_train, keep_prob: config['learn']['keep_prob']})
        train_track.increment(current_loss,train_score,train_classes_score)

        if iter != 0 and iter %config['common']['print_freq']==0:
            train_cum_loss,train_cum_score,train_classes_cum_score = train_track.average()
            train_cum_loss_std,train_cum_score_std,train_classes_cum_score_std = train_track.stdev()
            print( 'Iteration: {},\t Training loss: {:.4f}({:.4f}),\t Training DICE: {:.4f}({:.4f}),\t Per Class: {}({})\t'.format(iter, train_cum_loss, train_cum_loss_std, train_cum_score,train_cum_score_std, str(np.round(train_classes_cum_score,4)),str(np.round(train_classes_cum_score_std,4))))

            writer.add_summary(summary, sess.run(global_step))
            writer.flush()
            train_track = tracker.tracker(num_classes=config['data_train']['num_classes'])

        if iter != 0 and iter %config['learn']['val_freq']==0:

            val_track = tracker.tracker(num_classes=config['data_val']['num_classes'])
            for viter in range(np.floor(loader_val.__len__()/config['data_val']['batch_size']).astype(int)):
                try:
                    [ im_val, seg_val, dist_val, name_val ] = loader_val.fetch_batch()
                except:
                    print('fetch_batch on val set did not work')
                    continue
                val_summary, val_pred, val_loss, val_classes_score, val_score = sess.run([val_summary_op, pred, loss, score_classes, avg_score], feed_dict = {input: im_val, gt: seg_val, distance: dist_val, keep_prob:1.0})
                val_track.increment(val_loss,val_score,val_classes_score)
            val_cum_loss,val_cum_score,val_classes_cum_score = val_track.average()
            val_cum_loss_std,val_cum_score_std,val_classes_cum_score_std = val_track.stdev()

            print( 'Validation loss: {:.4f}({:.4f}),\t Validation DICE: {:.4f}({:.4f}),\t Per Class: {}({})\t'.format(val_cum_loss, val_cum_loss_std, val_cum_score,val_cum_score_std, str(np.round(val_classes_cum_score,4)),str(np.round(val_classes_cum_score_std,4))))
            writer.add_summary(val_summary, sess.run(global_step))
            writer.flush()

            save_flag, stop_flag = patience.track(val_cum_score)

            if save_flag:
                print( '!!!New checkpoint at step: {}\t, Validation DICE: {:.4f}\t'.format( iter, val_cum_score ) )
                checkpoint_path = config['common']['save_path']+sys.stdout.name+'/model.ckpt'
                saver.save(sess, checkpoint_path, global_step = global_step)

            if stop_flag:
                print('Stopping model due to no improvement for {} validation runs'.format(config['learn']['patience']) )
                writer.close()
                sess.close()
                break

    writer.close()
    print('Model finished training for {} steps'.format(iter))
