import numpy as np
import sys
import os
from os.path import join
import random
import h5py
import pickle
import scipy.io as spio
import scipy.io as sio
from skimage.util import view_as_blocks
from utils.augmentation import resize2D_me, resize3D_me, clip_me, rotate_me, flip_ud, flip_lr, flip_backforth, crop_volume, normalize_me
import pdb
from matplotlib import pyplot as plt
from PIL import Image
from scipy import misc
import imageio
import nibabel as nib
import pandas as pd

def load_h5py(fname):
    with h5py.File(fname,'r') as hf:
        img = np.array(hf['data'])
        if np.iscomplex(img).all():
            img = np.abs(img).astype('float32')
        else:
            img = img.astype('float32')
    return img

def load_h5py_flipped(fname):
    with h5py.File(fname,'r') as hf:
        img = np.array(hf['data_flipped'])
        if np.iscomplex(img).all():
            img = np.abs(img).astype('float32')
        else:
            img = img.astype('float32')
    return img

def load_h5py_maxval(fname):
    with h5py.File(fname,'r') as hf:
        maxval = np.array(hf['max']).astype('float32')
    return maxval

def load_png(fname):
    image = imageio.imread(fname).astype(np.float32)
    return image[...,0]

def load_raw(fname):
   raw_array = np.fromfile(fname, dtype=np.float32)
   im = np.reshape(raw_array, (500,500), order='C')
   assert im.shape == (500, 500), 'Found wrong shape: {}, {}'.format(fname, im.shape)
   return im

def load_nifti(filename):
    x = nib.load(filename)
    img = np.asanyarray(x.dataobj)
    img.astype(np.float32())
    return img

class data_loader_scapula:
    def __init__( self, data_root, batch_size, im_dims, crop, num_classes, idx_classes, num_channels, evaluate_mode=False, augment=False):
        # modify these lines to increase data loader flexibility
        if '.pkl' in data_root or '.pickle' in data_root:
            with open( data_root, "rb" ) as lf:
                list_files = pickle.load( lf )
        elif '.csv' in data_root:
            list_files = pd.read_csv(data_root,header=None).values.tolist()
        elif '.lsx' in data_root or '.xls' in data_root:
            list_files = pd.read_excel(data_root).values.tolist()
        else:
            print('Our dataLoader is not prepared to accept such input data, see its __init__')
            exit()
            return

        self.all_files       = list_files
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
        self.augment         = augment

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

        img = load_h5py(fname_img)

        if img.shape[0]!=self.im_dims[0] or img.shape[1]!=self.im_dims[1] or img.shape[2]!=self.im_dims[2]:
            img = crop_volume(img, self.crop)

        if self.evaluate_mode:
            img = img/(img.max()+1e-9)
        else:
            if random.random() > 0.5:
                try:
                    img = img/(load_h5py_maxval(fname_img)+1e-9)
                except:
                    img =  img/(random.randrange(np.int(img.max())-1000, np.int(img.max())+1000)+1e-9)
            else:
                img = img/(img.max()+1e-9)

        img /= (np.percentile(img,85)+1e-9)

        seg = load_h5py(fname_seg)
        if seg.shape[0]!=self.im_dims[0] or seg.shape[1]!=self.im_dims[1] or seg.shape[2]!=self.im_dims[2]:
            seg = crop_volume(seg, self.crop)

        if seg.ndim == len(self.im_dims):
            seg = seg[...,np.newaxis]

        seg = seg[...,self.idx_classes]

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
            try:
                self.seg_batch[i:] = seg
            except:
                self.seg_batch[i,:,:,:,0] = seg
        self.batch_cnt += 1

        if self.batch_cnt >= self.batch_max:
            self.batch_cnt = 0
            if self.evaluate_mode == False:
                self.__shuffle__()

        return self.im_batch, self.seg_batch, self.name_batch

class data_loader:
    def __init__( self, data_root, batch_size, im_dims, crop, num_classes, idx_classes, num_channels, evaluate_mode=False, augment=False):
        # modify these lines to increase data loader flexibility
        if '.pkl' in data_root or '.pickle' in data_root:
            with open( data_root, "rb" ) as lf:
                list_files = pickle.load( lf )
        elif '.csv' in data_root:
            list_files = pd.read_csv(data_root,header=None).values.tolist()
        elif '.lsx' in data_root or '.xls' in data_root:
            list_files = pd.read_excel(data_root).values.tolist()
        else:
            print('Our dataLoader is not prepared to accept such input data, see its __init__')
            exit()
            return

        self.all_files       = list_files
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
        self.augment         = augment

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

        img = load_nifti(fname_img)
        if img.shape[0]!=self.im_dims[0] or img.shape[1]!=self.im_dims[1] or img.shape[2]!=self.im_dims[2]:
            img = crop_volume(img, self.crop)
        img = img/img.max()
        img /= np.percentile(img,85)

        seg = load_nifti(fname_seg)
        if seg.shape[0]!=self.im_dims[0] or seg.shape[1]!=self.im_dims[1] or seg.shape[2]!=self.im_dims[2]:
            seg = crop_volume(seg, self.crop)

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
            try:
                self.seg_batch[i:] = seg
            except:
                self.seg_batch[i,:,:,:,0] = seg
        self.batch_cnt += 1

        if self.batch_cnt >= self.batch_max:
            self.batch_cnt = 0
            if self.evaluate_mode == False:
                self.__shuffle__()

        return self.im_batch, self.seg_batch, self.name_batch

class data_loader_nolabel:
    def __init__( self, data_root, batch_size, im_dims, crop, num_classes, idx_classes, num_channels, evaluate_mode=False, augment=False):
        # modify these lines to increase data loader flexibility
        if '.pkl' in data_root or '.pickle' in data_root:
            with open( data_root, "rb" ) as lf:
                list_files = pickle.load( lf )
        elif '.csv' in data_root:
            list_files = pd.read_csv(data_root,header=None).values.tolist()
        elif '.lsx' in data_root or '.xls' in data_root:
            list_files = pd.read_excel(data_root).values.tolist()
        else:
            print('Our dataLoader is not prepared to accept such input data, see its __init__')
            exit()
            return

        self.all_files       = list_files
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
        self.crop            = crop
        self.idx_classes     = idx_classes
        self.name_batch      = []
        self.augment         = augment

    def __len__( self ):
        return len(self.order_idx)

    def __shuffle__( self ):
        random.shuffle(self.order_idx)
        return self

    def __getitem__( self, key ):
        idx = self.order_idx[key]

        if len(self.all_files[0]) == 1 or len(self.all_files[0]) == 2 or len(self.all_files[0]) == 3:
            fname_img = self.all_files[idx][0]
        else:
            print('Our dataLoader is not prepared to accept such input data, see its __getitem__')
            exit()
            return

        img = load_nifti(fname_img)
        if img.shape[0]!=self.im_dims[0] or img.shape[1]!=self.im_dims[1] or img.shape[2]!=self.im_dims[2]:
            img = crop_volume(img, self.crop)
        img = img/img.max()
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
            if self.evaluate_mode == False:
                self.__shuffle__()

        return self.im_batch, self.name_batch

class data_loader_distance_nifti:
    def __init__( self, data_root, batch_size, im_dims, crop, num_classes, idx_classes, num_channels, evaluate_mode=False, augment=False):
        self.evaluate_mode = evaluate_mode
        if data_root[-3:] in ['pkl', 'kle']:
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


        self.all_files = list_files
        if self.evaluate_mode:
            self.order_idx       = list(range(len(self.all_files)))
        else:
            self.order_idx       = np.random.permutation(len(self.all_files))

        self.data_size       = len(self.order_idx)
        self.batch_size      = batch_size
        self.batch_cnt       = 0
        self.batch_max       = np.ceil(self.data_size/self.batch_size)
        self.im_dims         = im_dims
        self.im_batch        = np.zeros((self.batch_size, self.im_dims[0], self.im_dims[1], self.im_dims[2], num_channels), dtype='float32')
        self.seg_batch       = np.zeros((self.batch_size, self.im_dims[0], self.im_dims[1], self.im_dims[2], num_classes), dtype='uint8')
        self.dist_batch      = np.zeros((self.batch_size, self.im_dims[0], self.im_dims[1], self.im_dims[2], num_classes), dtype='float32')
        self.crop            = crop
        self.idx_classes     = idx_classes
        self.files_to_flip   = ''
        self.name_batch      = []
        self.augment         = augment
    def __len__( self ):
        return len(self.order_idx)

    def __shuffle__( self ):
        random.shuffle(self.order_idx)
        return self

    def __getitem__( self, key ):
        idx = self.order_idx[key]
        if len(self.all_files[0]) == 2:
            fname_img = self.all_files[idx][0]
            fname_seg = self.all_files[idx][1]
        elif len(self.all_files[0]) == 3:
            fname_img = self.all_files[idx][0]
            fname_seg = self.all_files[idx][1]
            fname_dist= self.all_files[idx][2]
        else:
            print('Our dataLoader is not prepared to accept such input data, see its __getitem__')
            exit()
            return

        img = load_nifti(fname_img)
        img[img<0]=0
        if img.shape[0]!=self.im_dims[0] or img.shape[1]!=self.im_dims[1] or img.shape[2]!=self.im_dims[2]:
            img = crop_volume(img, self.crop)
        img = img/img.max()
        img /= np.percentile(img,85)

        seg = load_nifti(fname_seg)
        if seg.shape[0]!=self.im_dims[0] or seg.shape[1]!=self.im_dims[1] or seg.shape[2]!=self.im_dims[2]:
            seg = crop_volume(seg, self.crop)

        dist = load_nifti(fname_dist)
        dist = 1.0+ dist
        if dist.shape[0]!=self.im_dims[0] or dist.shape[1]!=self.im_dims[1] or dist.shape[2]!=self.im_dims[2]:
            dist = crop_volume(dist, self.crop)

        if self.augment:
            if np.random.rand(1)>0.5:
                img = flip_backforth(img)
                seg = flip_backforth(seg)
                dist = flip_backforth(dist)
        return img, seg, dist, fname_img

    def fetch_batch(self):
        self.name_batch      = []
        for i in range(self.batch_size):
            if (i + self.batch_size*self.batch_cnt) < self.data_size:
                idx = self.order_idx[i + self.batch_size*self.batch_cnt]
            else:
                idx = self.order_idx[random.randint(0, self.data_size)]

            img, seg, dist, name = self.__getitem__( idx )
            if img.shape != self.im_dims:
                img_tmp = np.zeros(self.im_dims)
                img_tmp[...,:img.shape[-1]]= img
                img = img_tmp

            self.name_batch.append(name)
            if img.ndim == 3:
                self.im_batch[i,:,:,:,0] = img
                self.seg_batch[i,:,:,:,0] = seg
                self.dist_batch[i,:,:,:,0] = dist
            else:
                self.im_batch[i:] = img

            try: #for multiclass
                self.seg_batch[i:] = seg
                self.dist_batch[i:] = dist
            except: #for binary
                self.seg_batch[i,:,:,:,0] = seg
                self.dist_batch[i,:,:,:,0] = dist

        self.batch_cnt += 1

        if self.batch_cnt >= self.batch_max:
            self.batch_cnt = 0
            if self.evaluate_mode == False:
                self.__shuffle__()

        return self.im_batch, self.seg_batch, self.dist_batch, self.name_batch
