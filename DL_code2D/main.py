# We name this code after Dioscoridess. He was a Greek Physician who worked in Rome in the 1st Century A.D.
# author of De Materia Medica. He recommended to use ivy for treatment of OA.
# main codebase written by:
# Claudia Iriondo Claudia.Iriondo@ucsf.edu
# Francesco Caliva Francesco.Caliva@ucsf.edu
# to run use the syntax:
# python main.py --cfg path-to-yaml-cfg-file --desc name-to-use-when-you-save-the-model
import numpy as np
import os
import sys
sys.path.append('./utils')
import pdb
import argparse
import yaml
import tensorflow as tf

from logger import logger
from pprint import pprint
from utils import losses, early_stop, metrics, tb_vis, optimizers, tracker

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
    yaml_path = './cfg/train.yaml'
    desc = ''
    with open(yaml_path) as f:
        config = yaml.load(f)
    desc = 'BrainHack2020'

try:
    sys.path.append('./models/'+config['model_folder'])
except:
    sys.path.append('./models/'+config['model'])
import network as nn

try:
    import importlib
    dataLoader = importlib.import_module(config['dataLoader_folder'])
except:
    print('using except dataloader, please update yaml config file')
    import dataLoader as dataLoader

if not os.path.exists(config['common']['log_path']):
    os.makedirs(config['common']['log_path'])
if not os.path.exists(config['common']['save_path']):
    os.makedirs(config['common']['save_path'])

sys.stdout = logger(sys.stdout,path=config['common']['log_path'],desc=desc)

print('\n\n',sys.stdout.name,'\n\n')
pprint(config)
if 'all' not in config['common']['vis_GPU']:
    os.environ['CUDA_VISIBLE_DEVICES'] = config['common']['vis_GPU']

c = tf.ConfigProto()
c.gpu_options.allow_growth=True
c.gpu_options.per_process_gpu_memory_fraction = 0.95
c.allow_soft_placement = True
c.log_device_placement = False

seed = config['common']['seed']
tf.reset_default_graph()
tf.set_random_seed(seed)
model = nn.__dict__[config['model']](**config['model_params'])
global_step = tf.Variable(0, dtype=tf.int64, trainable=False)

input_shape = np.concatenate(([config['data_train']['batch_size']],config['data_train']['im_dims'],[config['data_train']['num_channels']]))
gt_shape = np.concatenate(([config['data_train']['batch_size']],config['data_train']['im_dims'],[config['data_train']['num_classes']]))

input = tf.placeholder(dtype=tf.float32, shape=input_shape)
gt = tf.placeholder(dtype=tf.uint8, shape=gt_shape)
keep_prob = tf.placeholder(dtype=tf.float32, shape=())

loader_train = dataLoader.__dict__[config['learn']['dataloader']](**config['data_train'])
loader_val = dataLoader.__dict__[config['learn']['dataloader']](**config['data_val'])
logits = model.network_fn(input,keep_prob)
weights = tf.constant([config['learn']['weights']], dtype='float32')

optimizer = optimizers.__dict__[config['learn']['optimizer']](config['learn']['lr'],global_step)
loss, pred = losses.__dict__[config['learn']['loss']](gt,logits,weights)

score_classes, avg_score = metrics.__dict__[config['learn']['metrics']](gt,pred)

trainer = optimizer.minimize(loss = loss, global_step = global_step)

if len(config['data_train']['im_dims']) == 3:
    train_summary_op = tb_vis.summarize3D(config['learn']['comp'],config['learn']['key_slice'],'train', input, gt, pred, config['data_train']['im_dims'][0],config['data_train']['im_dims'][1],loss,avg_score,score_classes)
    val_summary_op = tb_vis.summarize3D(config['learn']['comp'],config['learn']['key_slice'],'val',input, gt, pred, config['data_train']['im_dims'][0],config['data_train']['im_dims'][1],loss,avg_score,score_classes)
if len(config['data_train']['im_dims']) == 2:
    train_summary_op = tb_vis.summarize2D(config['learn']['comp'],config['learn']['key_slice'],'train', input, gt, pred, config['data_train']['im_dims'][0],config['data_train']['im_dims'][1],loss,avg_score,score_classes, optimizer._lr)
    val_summary_op = tb_vis.summarize2D(config['learn']['comp'],config['learn']['key_slice'],'val',input, gt, pred, config['data_train']['im_dims'][0],config['data_train']['im_dims'][1],loss,avg_score,score_classes, optimizer._lr)

init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

saver = tf.train.Saver(max_to_keep=5)

patience = early_stop.early_stop(patience=config['learn']['patience'])

# with tf.Session(config=c) as sess:
#     writer = tf.summary.FileWriter(config['common']['save_path']+sys.stdout.name,sess.graph)
#     init_op.run()
#     print('running')
#     train_track = tracker.tracker(num_classes=config['data_train']['num_classes'])
#     for iter in range(config['learn']['max_steps']):
#         im_train = np.random.normal(size=[4,512,512,1])
#         seg_train = np.random.normal(size=[4,512,512,3])
#         _, summary, current_loss, train_classes_score, train_score= sess.run([trainer, train_summary_op, loss, score_classes, avg_score], feed_dict = {input: im_train, gt: seg_train, keep_prob: config['learn']['keep_prob']})
#         print(optimizer._lr)
#         writer.add_summary(summary, sess.run(global_step))
#         writer.flush()
# pdb.set_trace()



with tf.Session(config=c) as sess:
    writer = tf.summary.FileWriter(config['common']['save_path']+sys.stdout.name,sess.graph)
    init_op.run()
    if config['pretrain']['flag']:
        model_name = config['pretrain']['ckpt']
        if "model.ckpt-" not in model_name:
            ckpt_id_available = ([x.split('model.ckpt-')[-1].split('.')[0] for x in os.listdir(model_name)])
            ckpt_id_available = np.max([np.int(x) for x in ckpt_id_available if(x!='events' and x!='checkpoint')])
            if model_name[-1]=='/':
                model_name = model_name+ 'model.ckpt-'+str(ckpt_id_available)
            else:
                model_name = model_name+ '/model.ckpt-'+str(ckpt_id_available)
        saver.restore(sess, model_name)
        print(f'Restore {model_name}')
        val_track = tracker.tracker(num_classes=config['data_val']['num_classes'])
        for viter in range(np.min((60,np.floor(loader_val.__len__()/config['data_val']['batch_size']).astype(int))).astype(int)):
            try:
                [ im_val, seg_val, name_val ] = loader_val.fetch_batch()
            except:
                loader_val.batch_cnt += 1
                print('fetch_batch on val set did not work')
                continue
            val_summary, val_loss, val_classes_score, val_score, val_pred = sess.run([val_summary_op, loss, score_classes, avg_score, pred], feed_dict = {input: im_val, gt: seg_val,keep_prob:1.0})
            val_track.increment(val_loss,val_score,val_classes_score)
        _,val_cum_score,_ = val_track.average()
        _, _ = patience.track(val_cum_score)
        print('Restored successfully. Initial score {}'.format(val_cum_score))
        # from scipy.io import savemat
        # savemat('/data/knee_mri8/Francesco/scapula_project_RAP/DL_code2D/validation_example',{'input':im_val,'gt':seg_val,'pred': val_pred})

    print('\n\n',config['common']['save_path'] + sys.stdout.name,'\n\n')
    print('Training')
    train_track = tracker.tracker(num_classes=config['data_train']['num_classes'])

    for iter in range(config['learn']['max_steps']):
        try:
            [ im_train, seg_train, name_train ] = loader_train.fetch_batch()
        except:
            loader_train.batch_cnt += 1
            print('fetch_batch on train set did not work')
            continue
        _, summary, current_loss, train_classes_score, train_score, train_pred= sess.run([trainer, train_summary_op, loss, score_classes, avg_score, pred], feed_dict = {input: im_train, gt: seg_train, keep_prob: config['learn']['keep_prob']})
        # savemat('/data/knee_mri8/Francesco/scapula_project_RAP/DL_code2D/training_example',{'input':im_train,'gt':seg_train,'pred': train_pred})
        # pdb.set_trace()
        train_track.increment(current_loss,train_score,train_classes_score)
        if iter != 0 and iter %config['common']['print_freq']==0:
            train_cum_loss,train_cum_score,train_classes_cum_score = train_track.average()
            train_cum_loss_std,train_cum_score_std,train_classes_cum_score_std = train_track.stdev()
            print( 'Iteration: {},\t Training loss: {:.4f}\u00B1{:.4f},\t Training score: {:.4f}\u00B1{:.4f},\t Per Class: {}\u00B1({})\t'.format(iter, train_cum_loss, train_cum_loss_std, train_cum_score,train_cum_score_std, str(np.round(train_classes_cum_score,4)),str(np.round(train_classes_cum_score_std,4))))
            writer.add_summary(summary, sess.run(global_step))
            writer.flush()
            train_track = tracker.tracker(num_classes=config['data_train']['num_classes'])

        if iter != 0 and iter %config['learn']['val_freq']==0:
            val_track = tracker.tracker(num_classes=config['data_val']['num_classes'])
            for viter in range(np.min((100,np.floor(loader_val.__len__()/config['data_val']['batch_size']).astype(int))).astype(int)):
                try:
                    [ im_val, seg_val, name_val ] = loader_val.fetch_batch()
                except:
                    loader_val.batch_cnt += 1
                    print('fetch_batch on val set did not work')

                    continue
                val_summary, val_loss, val_classes_score, val_score = sess.run([val_summary_op, loss, score_classes, avg_score], feed_dict = {input: im_val, gt: seg_val,keep_prob:1.0})

                val_track.increment(val_loss,val_score,val_classes_score)
            val_cum_loss,val_cum_score,val_classes_cum_score = val_track.average()
            val_cum_loss_std,val_cum_score_std,val_classes_cum_score_std = val_track.stdev()
            print( 'Validation loss: {:.4f}\u00B1{:.4f},\t Validation score: {:.4f}\u00B1{:.4f},\t Per Class: {}\u00B1({})\t'.format(val_cum_loss, val_cum_loss_std, val_cum_score,val_cum_score_std, str(np.round(val_classes_cum_score,4)),str(np.round(val_classes_cum_score_std,4))))
            writer.add_summary(val_summary, sess.run(global_step))
            writer.flush()

            save_flag, stop_flag = patience.track(val_cum_score)

            if save_flag:
                print( '!!!New checkpoint at step: {}\t, Validation score: {:.4f}\t'.format( iter, val_cum_score ) )
                checkpoint_path = config['common']['save_path']+sys.stdout.name+'/model.ckpt'
                saver.save(sess, checkpoint_path, global_step = global_step)

            if stop_flag:
                print('Stopping model due to no improvement for {} validation runs'.format(config['learn']['patience']) )
                writer.close()
                sess.close()
                break
    print('\n\n',config['common']['save_path'] + sys.stdout.name,'\n\n')
    writer.close()
    print('Model finished training for {} steps'.format(iter))
