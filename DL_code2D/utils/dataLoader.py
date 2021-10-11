import numpy as np
import sys
import os
from os.path import join
import h5py
import pickle
import random
from matplotlib import pyplot as plt
import time
import pandas as pd
from scipy.io import loadmat
import pdb

def load_h5py(fname, view='', slice=0):
    with h5py.File(fname,'r') as hf:
        img = np.array(hf['data'])
        if np.iscomplex(img).all():
            img = np.abs(img).astype('float32')
        else:
            img = img.astype('float32')
    if view == 'sagittal':
        img = img.transpose([0,2,1])
    elif view == 'coronal':
        img = img.transpose([2,1,0])
    return img[...,slice]

def load_mat(fname,flag):
    if flag == 'MRI':
        try:
            img = np.array(loadmat(fname)['im'])
        except:
            try:
                img = np.array(loadmat(fname)['this_slice'])
            except:
                img = np.array(loadmat(fname)['slice'])

    elif flag == 'mask':
        img = np.array(loadmat(fname)['label'])
    return img

show_example = False
if show_example:
    mri = load_h5py('/data/knee_mri8/Francesco/BrainHack/Data/train/9003815.im')
    file = load_h5py('/data/knee_mri8/Francesco/BrainHack/Data/train/9003815.seg')
    plt.imshow(mri[:,:,50],cmap='gray'),plt.title('MRI')
    plt.imshow(file[:,:,50,0]),plt.title('Background')
    plt.imshow(file[:,:,50,1]),plt.title('Femur')
    plt.imshow(file[:,:,50,2]),plt.title('Tibia')
    plt.imshow(file[:,:,50,3]),plt.title('Patella')
    plt.imshow(file[:,:,50,1] + file[:,:,50,2] + file[:,:,50,3]),plt.title('all')

def crop_volume(volume,crop):
    if crop[1] == 0:
        volume = volume[crop[0]:,...]
    else:
        volume = volume[crop[0]:-crop[1],...]

    if crop[3] == 0:
        volume = volume[:,crop[2]:,...]
    else:
        volume = volume[:,crop[2]:-crop[3],...]
    if len(crop)>4:
        if crop[5] == 0:
            volume = volume[:,:,crop[4]:,...]
        else:
            volume = volume[:,:,crop[4]:-crop[5],...]
    return volume

def get_normalization_values(normalization_file):
    list_files = pd.read_csv(normalization_file,header=None).values.tolist()
    minmax_vals = dict()
    for file in list_files:
        id_pat = str(file[0])
        minmax_vals[id_pat] = dict()
        minmax_vals[id_pat]['min']=file[1]
        minmax_vals[id_pat]['max']=file[2]
    return minmax_vals

class data_loader:
    def __init__( self, data_root, batch_size, im_dims, crop, num_classes, idx_classes, num_channels, normalization_file = ' ', evaluate_mode=False, view = ''):
        # modify these lines to increase data loader flexibility
        if view == 'coronal':
            nSlices = 364
        elif view == 'axial':
            nSlices = 124
        elif view == 'sagittal':
            nSlices = 364

        if '.pkl' in data_root or '.pickle' in data_root:
            with open( data_root, "rb" ) as lf:
                list_files = pickle.load( lf )

            lof = []
            counter_slices = []
            for ff in list_files:
                for c_sl in range(nSlices):
                    lof.append(ff)
                    counter_slices.append(c_sl)
            list_files = lof
        elif '.csv' in data_root:
            list_files = pd.read_csv(data_root,header=None).values.tolist()
        elif '.lsx' in data_root or '.xls' in data_root:
            list_files = pd.read_excel(data_root).values.tolist()
        else:
            print('Our dataLoader is not prepared to accept such input data, see its __init__')
            exit()
            return

        self.all_files       = list_files
        self.counter_slices   = counter_slices
        self.evaluate_mode  = evaluate_mode

        if self.evaluate_mode:
            self.order_idx       = list(range(len(self.all_files)))
        else:
            self.order_idx       = np.random.permutation(len(self.all_files))

        self.data_size       = len(self.order_idx)
        self.batch_size      = batch_size
        self.batch_cnt       = 0
        self.batch_max       = np.ceil(self.data_size/self.batch_size)
        self.im_dims         = im_dims
        self.im_batch        = np.zeros([self.batch_size]+[x for x in self.im_dims]+[num_channels], dtype='float32')
        self.seg_batch       = np.zeros([self.batch_size]+[x for x in self.im_dims]+[num_classes], dtype='uint8')
        self.crop            = crop
        self.idx_classes     = idx_classes
        self.name_batch      = []
        self.view            = view

        # if ' ' not in normalization_file:
        #     self.norm_values = get_normalization_values(normalization_file)
        #     self.normalize_input = True
        # else:
        #     self.norm_values = normalization_file
        #     self.normalize_input = False

    def __len__( self ):
        return len(self.order_idx)

    def __shuffle__( self ):
        random.shuffle(self.order_idx)
        return self

    def __getitem__( self, key ):
        idx = self.order_idx[key]
        if len(self.all_files[0]) == 2 or len(self.all_files[0]) == 3:
            fname_img = self.all_files[idx][0]
            fname_seg= self.all_files[idx][1]
        else:
            print('Our dataLoader is not prepared to accept such input data, see its __getitem__')
            exit()
            return
        if '.mat' in fname_img:
            img = load_mat(fname_img,'MRI')
        else:
            img = load_h5py(fname_img,view=self.view, slice= self.counter_slices[idx])

        if self.evaluate_mode:
            img /= (img.max()+1e-9)
        else:
            img =  img/(random.randrange(np.int(img.max())-500, np.int(img.max())+500)+1e-9)

        img = crop_volume(img, self.crop)
        if len(self.im_dims)==3 or len(self.im_dims)==2:
            img /= (np.percentile(img,85)+1e-9)

        if '.mat' in fname_seg:
            seg = load_mat(fname_seg,'mask')
        else:
            seg = load_h5py(fname_seg,view=self.view, slice= self.counter_slices[idx]).astype('uint8')

#            with h5py.File(fname_seg,'r') as hf:
#                seg = np.array(hf['data']).astype('uint8')

        if seg.ndim == len(self.im_dims):
            seg = seg[...,np.newaxis]

        seg = seg[...,self.idx_classes]

        seg = crop_volume(seg, self.crop)

        return img, seg, fname_img

    def fetch_batch(self):
        self.name_batch = []
        for i in range(self.batch_size):
            if (i + self.batch_size*self.batch_cnt) < self.data_size:
                idx = self.order_idx[i + self.batch_size*self.batch_cnt]
            else:
                idx = self.order_idx[random.randint(0, self.data_size)]

            img, seg, name = self.__getitem__( idx )
            self.name_batch.append(name)
            if img.ndim == len(self.im_dims):
                self.im_batch[i,...,0] = img
            else:
                self.im_batch[i:] = img
            self.seg_batch[i:] = seg

        self.batch_cnt += 1

        if self.batch_cnt >= self.batch_max:
            self.batch_cnt = 0
            if self.evaluate_mode == False:
                self.__shuffle__()

        return self.im_batch, self.seg_batch, self.name_batch


class data_loader_infer_nolabel:
    def __init__( self, data_root, batch_size, im_dims, crop, num_classes, idx_classes, num_channels, normalization_file = ' '):
        # modify these lines to increase data loader flexibility

        if '.pkl' in data_root or '.pickle' in data_root:
            with open( data_root, "rb" ) as lf:
                list_files = pickle.load( lf )
        elif '.csv' in data_root:
            list_files = pd.read_csv(data_root, header=None).values.tolist()
        elif '.lsx' in data_root or '.xls' in data_root:
            list_files = pd.read_excel(data_root).values.tolist()
        else:
            print('Our dataLoader is not prepared to accept such input data, see its __init__')
            exit()
            return

        self.all_files       = list_files
        self.order_idx       = list(range(len(self.all_files)))
        self.data_size       = len(self.order_idx)
        self.batch_size      = batch_size
        self.batch_cnt       = 0
        self.batch_max       = np.ceil(self.data_size/self.batch_size)
        self.im_dims         = im_dims

        self.im_batch        = np.zeros([self.batch_size]+[x for x in self.im_dims]+[num_channels], dtype='float32')
        self.crop            = crop
        self.idx_classes     = idx_classes
        self.name_batch      = []

        if ' ' not in normalization_file:
            self.norm_values = get_normalization_values(normalization_file)
            self.normalize_input = True
        else:
            self.norm_values = normalization_file
            self.normalize_input = False

    def __len__( self ):
        return len(self.order_idx)

    def __getitem__( self, key ):
        idx = self.order_idx[key]
        fname_img = self.all_files[idx]
        if isinstance(fname_img,list):
            fname_img = fname_img[0]
        if '.mat' in fname_img:
            img = load_mat(fname_img,'MRI')
        else:
            img = load_h5py(fname_img)
        try:
            if ' ' not in self.norm_values:
                idpat = fname_img.split('/')[-2]
                img = img- self.norm_values[idpat]['min']
                img = img/ (self.norm_values[idpat]['max']-self.norm_values[idpat]['min'])
        except:
            print(f'missing norm value for {idpat}')

        img = crop_volume(img, self.crop)
        if len(self.im_dims)==3 or len(self.im_dims)==2:
            img /= np.percentile(img,85)

        return img, fname_img

    def fetch_batch(self):
        self.name_batch = []
        for i in range(self.batch_size):
            if (i + self.batch_size*self.batch_cnt) < self.data_size:
                idx = self.order_idx[i + self.batch_size*self.batch_cnt]
            else:
                idx = self.order_idx[random.randint(0, self.data_size)]
            img, name = self.__getitem__( idx )
            self.name_batch.append(name)
            if img.ndim == len(self.im_dims):
                self.im_batch[i,...,0] = img
            else:
                self.im_batch[i:] = img
        self.batch_cnt += 1
        if self.batch_cnt >= self.batch_max:
            self.batch_cnt = 0
        return self.im_batch, self.name_batch
