# We name this code after Dioscoridess. He was a Greek Physician who worked in Rome in the 1st Century A.D.
# author of De Materia Medica. He recommended to use ivy for treatment of OA.
# main codebase written by:
# Claudia Iriondo Claudia.Iriondo@ucsf.edu
# Francesco Caliva Francesco.Caliva@ucsf.edu
#
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
    parser.add_argument('--gpu', default="")

    args = parser.parse_args()
    with open(args.cfg) as f:
         config = yaml.load(f, Loader=yaml.UnsafeLoader)
    desc = args.desc
else:
    cd /data/bigbone5/JacobOeding/DL_code
    yaml_path = '/data/bigbone5/JacobOeding/DL_code/cfgs/infer/infer_sagittal.yaml'
    desc = ''
    with open(yaml_path) as f:
        config = yaml.load(f)
    desc = 'infer_sagittal'

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

sys.stdout = logger(sys.stdout,path=config['common']['log_path'],desc=desc)
print('\n\n',sys.stdout.name,'\n\n')
pprint(config)

if 'all' not in config['common']['vis_GPU']:
    if args.gpu == "":
        os.environ['CUDA_VISIBLE_DEVICES'] = config['common']['vis_GPU']
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

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

input_shape = np.concatenate(([config['data_infer']['batch_size']],config['data_infer']['im_dims'],[config['data_infer']['num_channels']]))
gt_shape = np.concatenate(([config['data_infer']['batch_size']],config['data_infer']['im_dims'],[config['data_infer']['num_classes']]))

input = tf.placeholder(dtype=tf.float32, shape=input_shape)
gt = tf.placeholder(dtype=tf.uint8, shape=gt_shape)
keep_prob = tf.placeholder(dtype=tf.float32, shape=())

loader_infer = dataLoader.__dict__[config['learn']['dataloader']](**config['data_infer'])

logits = model.network_fn(input,keep_prob)

loss, pred = losses.__dict__[config['learn']['loss']](gt,logits)
score_classes, avg_score = metrics.__dict__[config['learn']['metrics']](gt,pred)

init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

saver = tf.train.Saver(max_to_keep=25)
if config['learn']['save_pred']:
    pred_path = config['common']['pred_path'] + sys.stdout.name + '/'
    os.makedirs(pred_path)

print('\n\n',config['common']['pred_path'] + sys.stdout.name,'\n\n')
with tf.Session() as sess:
    init_op.run()
    model_name = config['trained_model']['ckpt']
    if "model.ckpt-" not in model_name:
        ckpt_id_available = ([x.split('model.ckpt-')[-1].split('.')[0] for x in os.listdir(model_name)])
        ckpt_id_available = np.max([np.int(x) for x in ckpt_id_available if(x!='events' and x!='checkpoint')])
        model_name = model_name+ '/model.ckpt-'+str(ckpt_id_available)
    saver.restore(sess, model_name)
    print(f'Restore {model_name}')
    inf_track = tracker.tracker(num_classes=config['data_infer']['num_classes'])
    if len(config['data_infer']['im_dims'])==3:
        for viter in range(loader_infer.__len__()):
            [ im_inf, seg_inf, name_inf ] = loader_infer.fetch_batch()

            inf_loss, inf_classes_score, inf_score, inf_logits, inf_pred = sess.run([loss, score_classes, avg_score, logits, pred], feed_dict = {input: im_inf, gt: seg_inf, keep_prob: 1.0})
            inf_track.increment(inf_loss, inf_score, inf_classes_score)
            print('File: {}, Score: {:.4f}, Per Class: {}'.format(name_inf[0].split('/')[-1], inf_score, str(np.round(inf_classes_score,4))))
            if config['learn']['save_pred']:
                for i_cnt_save in range(im_inf.shape[0]):
                    filename = pred_path + name_inf[i_cnt_save].split('/')[-1].split('.')[0]
                    savemat(filename,{'input': im_inf[i_cnt_save,:], 'gt': seg_inf[i_cnt_save,:], 'pred': inf_pred[i_cnt_save,:]})
    elif len(config['data_infer']['im_dims'])==2:
        if config['data_infer']['view'] == 'coronal':
            volSize = 364
        elif config['data_infer']['view'] == 'axial':
            volSize = 124
        elif config['data_infer']['view'] == 'sagittal':
            volSize = 364
        elif config['data_infer']['view'] == 'sagittal_nikan':
            volSize = 160
        elif config['data_infer']['view'] == 'axial_nikan':
            volSize = 384 #302

        current_name = ''
        jj=-1
        for viter in range(np.floor(loader_infer.__len__()/config['data_infer']['batch_size']).astype(int)):
            try:
                [ im_inf, seg_inf, name_inf ] = loader_infer.fetch_batch()
            except:
                loader_infer.batch_cnt += 1
                print('file corrupt')
                continue

            volname = name_inf[0].split('/')[-1].split('_I')[0].split('.im')[0]

            if volname in current_name:
                new_volume = False
            else:
                if jj != -1:
                    filename = pred_path + current_name
                    if ("nikan" in config['data_infer']['view']) or ("axial_nikan" in config['data_infer']['view']):
                        # savemat(filename,{'input': vol_im, 'pred':vol_pred.astype(np.float16)})
                        savemat(filename,{'pred':vol_pred.astype(np.float16)})
                    else:
                        savemat(filename,{'input': vol_im.astype(np.float16), 'gt': vol_seg.astype(np.uint8), 'pred':vol_pred.astype(np.float16)})
                    print(f'processed {filename}')
                current_name = volname
                new_volume = True
                jj=0
            if new_volume:
                vol_im   = np.zeros((np.hstack((config['data_infer']['im_dims'],volSize)).tolist()))
                vol_seg  = np.zeros((np.hstack((config['data_infer']['im_dims'],volSize,config['data_infer']['num_classes'])).tolist()))
                vol_pred = np.zeros((np.hstack((config['data_infer']['im_dims'],volSize,config['data_infer']['num_classes'])).tolist()))

            inf_pred, inf_loss, inf_classes_score, inf_score, inf_logits = sess.run([pred, loss, score_classes, avg_score, logits], feed_dict = {input: im_inf, gt: seg_inf,keep_prob:1.0})

            for id_in_batch in range(config['data_infer']['batch_size']):
                idx_vol = jj#(viter*config['data_infer']['batch_size'] + id_in_batch)%volSize
                try:
                    vol_im[...,idx_vol]   = np.squeeze(im_inf[id_in_batch,...])
                except:
                    break
                vol_seg[...,idx_vol,:]  = seg_inf[id_in_batch,...]
                vol_pred[...,idx_vol,:] = inf_pred[id_in_batch,...]
                jj+=1
                if np.sum(seg_inf[:,:,:,0:1])!=0:
                    inf_track.increment(inf_loss, inf_score, inf_classes_score)
        # add this point one volume is missing to be saved. Save it below
        #if viter == np.floor(loader_infer.__len__()/config['data_infer']['batch_size']).astype(int)-1:
        filename = pred_path + current_name
        # for sl in range(volSize):
        #     if np.all(sum(vol_im[...,sl])==0):
        #             vol_im = np.delete(vol_im,range(sl,volSize),axis=-1)
        #             vol_seg = np.delete(vol_seg,range(sl,volSize),axis=-2)
        #             vol_pred = np.delete(vol_pred,range(sl,volSize),axis=-2)
        #             break
        if "nikan" in config['data_infer']['view']:
            # savemat(filename,{'input': vol_im, 'pred':vol_pred.astype(np.float16)})
            savemat(filename,{'pred':vol_pred.astype(np.float16)})
        else:
            savemat(filename,{'input': vol_im.astype(np.float16), 'gt': vol_seg.astype(np.uint8), 'pred':vol_pred.astype(np.float16)})
        print(f'processed {filename}')


                # print('File: {}, Score: {:.4f}, Per Class: {}'.format(name_inf[0].split('/')[-1], inf_score, str(np.round(inf_classes_score,4))))
                # idx_vol = viter*config['data_infer']['batch_size'] + id_in_batch
                # pdb.set_trace()

                    # savemat(filename,{'pred':(vol_pred>=0.5).astype(np.uint8)})

#                     vol_im   = np.zeros((np.hstack((config['data_infer']['vol_size'],config['data_infer']['im_dims'])).tolist()))
#                     vol_seg  = np.zeros((np.hstack((config['data_infer']['vol_size'],config['data_infer']['im_dims'])).tolist()))
#                     vol_pred = np.zeros((np.hstack((config['data_infer']['vol_size'],config['data_infer']['im_dims'])).tolist()))
# ###





    inf_cum_loss, inf_cum_score, inf_classes_cum_score = inf_track.average()
    inf_cum_loss_std, inf_cum_score_std, inf_classes_cum_score_std = inf_track.stdev()
    print ('--'*15+'Summary'+'--'*15)
    print( 'Inference Score: {:.4f}\u00B1{:.4f}),\t Per Class: {}\u00B1({})\t'.format(inf_cum_score,inf_cum_score_std, str(np.round(inf_classes_cum_score,4)),str(np.round(inf_classes_cum_score_std,4))))
    print('\n\n',config['common']['pred_path'] + sys.stdout.name,'\n\n')
    start_time = time.time()
    for_timing = sess.run([pred], feed_dict = {input: im_inf, gt: seg_inf,keep_prob:1.0})
    elapsed_time = time.time() - start_time
    print('Single computation required {:.0f}m {:.0f}s'.format(elapsed_time // 60, elapsed_time % 60))

    sess.close()
