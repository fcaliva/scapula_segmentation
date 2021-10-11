import numpy as np
import keras
from keras_preprocessing.image import apply_affine_transform
from scipy.ndimage import zoom
from PIL import Image as Image

def clip_me(x,max_clip=1.1, min_clip=0):
   on_switch = np.random.randint(2, size=2)
   clip_range = random.uniform(min_clip,max_clip)
   clipped = np.clip(x, on_switch[0]*np.percentile(x,clip_range), on_switch[1]*np.percentile(x,100-clip_range)+int(not on_switch[1])*100)
   return clipped

def rotate_me(x,y, rg=40, row_axis=0,
                col_axis=1, channel_axis=2,
                fill_mode='nearest', cval=0.,
                interpolation_order=1):
    theta = np.random.uniform(-rg, rg)
    expanded_x = False
    if x.ndim < 3:
        x = x[...,np.newaxis]
        expanded_x = True
    x = apply_affine_transform(x, theta=theta, channel_axis=channel_axis,
                               fill_mode=fill_mode, cval=cval,
                               order=interpolation_order)
    y = apply_affine_transform(y, theta=theta, channel_axis=channel_axis,
                               fill_mode=fill_mode, cval=cval,
                               order=interpolation_order)
    if expanded_x is True:
        x = np.squeeze(x)
    y[y>0]=1
    y[y<0]=0
    return x, y

def flip_ud(x):
    return np.flip(x,axis=0)

def flip_lr(x):
    return np.flip(x,axis=1)

def flip_backforth(x):
    return np.flip(x,axis=2)

def crop_volume(volume,crop):
    if crop[1] == 0:
        volume = volume[crop[0]:,...]
    else:
        volume = volume[crop[0]:-crop[1],...]

    if len(crop) in [4,6]  :
        if crop[3] == 0:
            volume = volume[:,crop[2]:,...]
        else:
            volume = volume[:,crop[2]:-crop[3],...]

    if len(crop) == 6:
        if crop[5] == 0:
            volume = volume[:,:,crop[4]:,...]
        else:
            volume = volume[:,:,crop[4]:-crop[5],...]

    return volume

def crop_actual_image(image,label):
    images_t = []
    labels_t = []

    non_zero_y = np.where(np.sum(image, axis=0) > 0)
    non_zero_x = np.where(np.sum(image, axis=1) > 0)

    image_t = image[np.min(non_zero_x):np.max(non_zero_x),
                   np.min(non_zero_y):np.max(non_zero_y)]
    label_t = label[np.min(non_zero_x):np.max(non_zero_x),
                   np.min(non_zero_y):np.max(non_zero_y)]

    images_t.append(image_t)
    labels_t.append(label_t)

    image = np.array(images_t)
    label = np.array(labels_t)
    return image, label

def normalize_me(img):
    # return (((img - img[img>0].min()) * (1/(img.max() - img[img>0].min())))).astype('float32')
    return (((img - img.min()) * (1/(img.max() - img.min())))).astype('float32')

def resize2D_me(image, size):
    output = Image.fromarray(image).resize(size = size, resample = Image.BICUBIC)
    # output2 = np.array(resize(image, (size[0], size[1]), preserve_range=True) , dtype=image.dtype)
    return np.array(output, dtype=image.dtype)

def resize3D_me(volume, scale_factor  = (1,1,1)):

    return np.array(zoom(volume, scale_factor), dtype=volume.dtype)


def stretch_me(image,label):
    scale_x = random.randint(6,14)/10
    scale_y = random.randint(6,14)/10

    image_t = ndimg.affine_transform(image,(scale_x,scale_y), order=3,mode='constant').astype(np.float32)
    label_t = ndimg.affine_transform(label,(scale_x,scale_y), order=0,mode='constant').astype(np.float32)

    return np.array(images_t), np.array(labels_t)

def elasticDef(Data):
    points = random.randint(1,3)
    sigma = random.randint(6,12)
    [image_t, label_t] = elasticdeform.deform_random_grid([image, label], order=[3,0], sigma=sigma, points=points,axis=(0, 1),mode='constant')

    images_t.append(image_t.astype(np.float32))
    labels_t.append(label_t.astype(np.float32))

    return np.array(images_t), np.array(labels_t)

def shift_me (image, label):
    del_x = random.randint(-50,50)
    del_y = random.randint(-50,50)

    image_t = ndimg.shift(image,shift=(del_x,del_y), order=3,cval=0.0).astype(np.float32)
    label_t = ndimg.shift(label,shift=(del_x,del_y), order=0,cval=0.0).astype(np.float32)

    return np.array(images_t), np.array(labels_t)

def intensity_augmentation(image):
    del_x = random.randint(-100,100)
    del_y = random.randint(-100,100)
    gauss = ndimg.shift(gkern(),shift=(del_x,del_y), order=3,cval=1.0).astype(np.float32)
    return image*gauss
