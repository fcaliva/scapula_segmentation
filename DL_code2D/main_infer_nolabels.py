# We name this code after Dioscoridess. He was a Greek Physician who worked in Rome in the 1st Century A.D.
# author of De Materia Medica. He recommended to use ivy for treatment of OA.
# main codebase written by:
# Claudia Iriondo Claudia.Iriondo@ucsf.edu
# Francesco Caliva Francesco.Caliva@ucsf.edu
import numpy as np
import os
import sys
sys.path.append('./utils')
import pdb
import argparse
import yaml
import time
import tensorflow as tf

from logger import logger
from pprint import pprint

from utils import metrics, losses, tracker
from scipy.io import savemat
trial = False
# pwd
# cd /data/knee_mri8/Francesco/BrainHack
if not trial:
    parser = argparse.ArgumentParser(description='define configuration file and run description')
    parser.add_argument('--cfg')
    parser.add_argument('--desc')
    args = parser.parse_args()
    with open(args.cfg) as f:
         config = yaml.load(f, Loader=yaml.UnsafeLoader)
    desc = args.desc
else:
    yaml_path = './cfg/infer_nolabel.yaml'
    desc = ''
    with open(yaml_path) as f:
        config = yaml.load(f)
    desc = 'BrainHack2020_infer_nolabel'

sys.path.append('./models/'+config['model'])
print('\n\n',sys.stdout.name,'\n\n')
pprint(config)
import network as nn

try:
    import importlib
    dataLoader = importlib.import_module(config['dataLoader_folder'])
except:
    print('using except dataloader, please update yaml config file')
    import dataLoader as dataLoader

if not os.path.exists(config['common']['log_path']):
    os.makedirs(config['common']['log_path'])

sys.stdout = logger(sys.stdout,path=config['common']['log_path'],desc=desc)
print('\n\n',sys.stdout.name,'\n\n')
if 'all' not in config['common']['vis_GPU']:
    os.environ['CUDA_VISIBLE_DEVICES'] = config['common']['vis_GPU']


c = tf.ConfigProto()

seed = config['common']['seed']
tf.reset_default_graph()
tf.set_random_seed(seed)
model = nn.__dict__[config['model']](**config['model_params'])

global_step = tf.Variable(0, dtype=tf.int64, trainable=False)

input_shape = np.concatenate(([config['data_infer']['batch_size']],config['data_infer']['im_dims'],[config['data_infer']['num_channels']]))

input = tf.placeholder(dtype=tf.float32, shape=input_shape)
keep_prob = tf.placeholder(dtype=tf.float32, shape=())

loader_infer = dataLoader.__dict__[config['learn']['dataloader']](**config['data_infer'])

logits = model.network_fn(input,keep_prob)
pred = tf.math.softmax(logits)

init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

saver = tf.train.Saver(max_to_keep=25)

if config['learn']['save_pred']:
    pred_path = config['common']['pred_path'] + sys.stdout.name + '/'
    os.makedirs(pred_path)

with tf.Session() as sess:
    init_op.run()
    saver.restore(sess, config['trained_model']['ckpt'])
    if len(config['data_infer']['im_dims'])==3:
        for viter in range(loader_infer.__len__()):
            [ im_inf, name_inf ] = loader_infer.fetch_batch()
            inf_pred = sess.run([pred], feed_dict = {input: im_inf, keep_prob: 1.0})

            print('File: {}, successfully processed'.format(name_inf[0]))
            if config['learn']['save_pred']:
                for i_cnt_save in range(im_inf.shape[0]):
                    filename = pred_path + name_inf[i_cnt_save].split('/')[-1].split('.')[0]
                    savemat(filename,{'input': np.squeeze(im_inf[i_cnt_save,:]), 'pred': np.squeeze(inf_pred[0][i_cnt_save,:])})
    elif len(config['data_infer']['im_dims'])==2:
        current_name = ''
        jj=-1
        for viter in range(np.floor(loader_infer.__len__()/config['data_infer']['batch_size']).astype(int)):
            try:
                [ im_inf, name_inf ] = loader_infer.fetch_batch()
            except:
                print(f'Loading error')
                loader_infer.batch_cnt += 1
                jj+=1
                continue

            if len(name_inf[0].split('/')[-1].split('_I'))>1:
                volname = name_inf[0].split('/')[-1].split('_I')[0]
            elif len(name_inf[0].split('/')[-1].split('_contra_'))>1:
                volname = name_inf[0].split('/')[-1].split('_contra_')[0]+'_contra'
            elif len(name_inf[0].split('/')[-1].split('_ipsi_'))>1:
                volname = name_inf[0].split('/')[-1].split('_ipsi_')[0]+'_ipsi'
            else:
                volname = ''.join(name_inf[0].split('/')[:-1])
            if volname in current_name:
                new_volume = False
            else:
                if jj != -1:
                    filename = pred_path + current_name
                    for sl in range(248):
                        if np.all(sum(vol_im[...,sl])==0):
                                vol_im = np.delete(vol_im,range(sl,volSize),axis=-1)
                                vol_pred = np.delete(vol_pred,range(sl,volSize),axis=-2)
                                break
                    savemat(filename,{'input': vol_im, 'pred':(vol_pred>=0.5).astype(np.uint8)})
                current_name = volname
                new_volume = True
                jj=0
            if new_volume:
                volSize = 600
                vol_im   = np.zeros((np.hstack((config['data_infer']['im_dims'],volSize)).tolist()))
                vol_pred = np.zeros((np.hstack((config['data_infer']['im_dims'],volSize,config['data_infer']['num_classes'])).tolist()))

            try:
                inf_pred = sess.run(pred, feed_dict = {input: im_inf, keep_prob: 1.0})
            except:
                print(f'Inference error with {name_inf}')
                jj+=1
                continue
            for id_in_batch in range(config['data_infer']['batch_size']):
                idx_vol = jj#(viter*config['data_infer']['batch_size'] + id_in_batch)%volSize
                # print(f'{id_in_batch}-{jj}')

                vol_im[...,idx_vol]   = np.squeeze(im_inf[id_in_batch,...])
                vol_pred[...,idx_vol,:] = np.squeeze(inf_pred[id_in_batch,...])
                jj+=1
        # add this point one volume is missing to be saved. Save it below
        #if viter == np.floor(loader_infer.__len__()/config['data_infer']['batch_size']).astype(int)-1:
        filename = pred_path + current_name
        for sl in range(600):
            if np.all(sum(vol_im[...,sl])==0):
                    vol_im = np.delete(vol_im,range(sl,volSize),axis=-1)
                    vol_pred = np.delete(vol_pred,range(sl,volSize),axis=-2)
                    break
        savemat(filename,{'input': vol_im, 'pred':(vol_pred>=0.5).astype(np.uint8)})

    print ('--'*15+'Summary'+'--'*15)
    print( 'Inference complete')
    start_time = time.time()
    for_timing = sess.run([pred], feed_dict = {input: im_inf,keep_prob:1.0})
    elapsed_time = time.time() - start_time
    print('Single computation required {:.0f}m {:.0f}s'.format(elapsed_time // 60, elapsed_time % 60))
    sess.close()
