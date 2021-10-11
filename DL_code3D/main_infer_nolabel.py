# DioscoriDESS was Greek Physician who worked in Rome in the 1st Century A.D.
# author of De Materia Medica. He recommended to use ivy for treatment of OA.
# main codebase written by Claudia and Francesco, slack us with any questions
import numpy as np
import os
import sys
sys.path.append('./utils')
import pdb
import argparse
import yaml
import time
import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import dataLoader_main as dataLoader

from logger import logger
from pprint import pprint

from utils import metrics, losses, tracker
from scipy.io import savemat

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
    yaml_path = '/data/knee_mri4/segchal_fra/Dioscorides/cfgs/inference/scapola/experiment2.yaml'
    desc = 'exp2'
    with open(yaml_path) as f:
        config = yaml.load(f)

sys.path.append('./models/'+config['model'])
import network as nn

if not os.path.exists(config['common']['log_path']):
    os.makedirs(config['common']['log_path'])

sys.stdout = logger(sys.stdout,path=config['common']['log_path'],desc=desc)
print('\n\n',sys.stdout.name,'\n\n')
pprint(config)
if 'all' not in config['common']['vis_GPU']:
    os.environ['CUDA_VISIBLE_DEVICES'] = config['common']['vis_GPU']
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.logging.set_verbosity(tf.logging.ERROR)

c = tf.ConfigProto()
c.gpu_options.allow_growth=True
c.gpu_options.per_process_gpu_memory_fraction = 0.95
c.allow_soft_placement = True
c.log_device_placement = False

seed = config['common']['seed']
np.random.seed(seed)
tf.reset_default_graph()
tf.set_random_seed(seed)
model = nn.__dict__[config['model']](**config['model_params'])

global_step = tf.Variable(0, dtype=tf.int64, trainable=False)

input_shape = np.concatenate(([config['data_infer']['batch_size']],config['data_infer']['im_dims'],[config['data_infer']['num_channels']]))

input = tf.placeholder(dtype=tf.float32, shape=input_shape)
keep_prob = tf.placeholder(dtype=tf.float32, shape=())

loader_infer = dataLoader.__dict__[config['learn']['dataloader']](**config['data_infer'])

logits = model.network_fn(input,keep_prob)

if config['learn']['activation'] == 'sigmoid':
    pred = tf.sigmoid(logits)
elif config['learn']['activation'] == 'softmax':
    pred = tf.math.softmax(logits)
elif config['learn']['activation'] == 'linear':
    pred = logits
init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

saver = tf.train.Saver(max_to_keep=25)

if config['learn']['save_pred']:
    pred_path = config['common']['pred_path'] + sys.stdout.name + '/'
    os.makedirs(pred_path)

with tf.Session() as sess:
    init_op.run()
    saver.restore(sess, config['trained_model']['ckpt'])
    # 3D data
    if len(config['data_infer']['im_dims'])==3:
        for viter in range(np.floor(loader_infer.__len__()/config['data_infer']['batch_size']).astype(int)):
            try:
                [im_inf, name_inf ] = loader_infer.fetch_batch()
            except:
                continue
            inf_pred = sess.run([pred], feed_dict = {input: im_inf, keep_prob: 1.0})
            inf_pred = np.asarray(inf_pred)

            if config['learn']['save_pred']:
                for i_cnt_save in range(im_inf.shape[0]):
                    ff = name_inf[i_cnt_save].split('/')[-1].split('.')[0]
                    print('File: {}, done.'.format(ff))
                    filename = pred_path + ff
                    savemat(filename,{'input': im_inf[i_cnt_save,:,:,:,0], 'pred': inf_pred[i_cnt_save,:,:,:,:]})
    # 2D_data:
    elif len(config['data_infer']['im_dims'])==2:
        vol_im   = np.zeros((np.hstack((config['data_infer']['vol_size'],config['data_infer']['im_dims'])).tolist()))
        vol_pred = np.zeros((np.hstack((config['data_infer']['vol_size'],config['data_infer']['im_dims'])).tolist()))
        for viter in range(np.floor(loader_infer.__len__()/config['data_infer']['batch_size']).astype(int)):
            [ im_inf, name_inf ] = loader_infer.fetch_batch()
            inf_pred = sess.run([pred], feed_dict = {input: im_inf, keep_prob:1.0})

            # idx_vol = viter%config['data_infer']['vol_size']
            for id_in_batch in range(config['data_infer']['batch_size']):
                idx_vol = (viter*config['data_infer']['batch_size'] + id_in_batch)%config['data_infer']['vol_size']
                vol_im[idx_vol,...]   = np.squeeze(im_inf[id_in_batch,...])
                vol_pred[idx_vol,...] = np.squeeze(inf_pred[id_in_batch,...])

            if viter != 0 and idx_vol%(config['data_infer']['vol_size']-1) == 0:
                filename = pred_path + name_inf[0].split('/')[-1].split('_slice_')[0]
                # savemat(filename,{'input': vol_im, 'gt': vol_seg.astype(np.uint8), 'pred':(vol_pred>=0.5).astype(np.uint8)})
                savemat(filename,{'im':vol_im, 'pred':(vol_pred>=0.5).astype(np.uint8)})

                vol_im   = np.zeros((np.hstack((config['data_infer']['vol_size'],config['data_infer']['im_dims'])).tolist()))
                vol_pred = np.zeros((np.hstack((config['data_infer']['vol_size'],config['data_infer']['im_dims'])).tolist()))
    print ('--'*15+'--'*15)
    start_time = time.time()
    for_timing = sess.run([pred], feed_dict = {input: im_inf, keep_prob:1.0})
    elapsed_time = time.time() - start_time
    print('Single computation required {:.0f}m {:.0f}s'.format(elapsed_time // 60, elapsed_time % 60))

    sess.close()
