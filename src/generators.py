import itertools
import random

import numpy as np
# Used for model training.
def mri_data_generator_scan_to_scan_train_groupReg(x_data, time ,norm, data_seg, batch_size=1, mode = 'train',p=None,s=None):
    """
    x_data[B,H,W,T,S];  [160,160,16] P=168 * S=5 = 840 combinations
    time[B,T,S]
    """
    # preliminary sizing
    vol_shape = x_data[0,:,:,:,0].shape # extract data shape
    ndims = len(vol_shape) - 1

    # prepare a zero array the size of the deformation
    # we'll explain this below
    zeros = np.zeros([batch_size, *vol_shape, ndims])

    while True:
        # prepare inputs:

        if mode == 'train':
            p = np.random.uniform(low=0,high=x_data.shape[0]-0.0001,size=batch_size)
            s = np.random.uniform(low=0,high=x_data.shape[4]-0.0001,size=batch_size)
        elif mode == 'test':
            p = np.arange(0,batch_size)
            s = np.random.uniform(low=2,high=2,size=batch_size)
            
        p = p.astype(int)
        s = s.astype(int)

        group_images = x_data[p, :, :,:, s, np.newaxis]
        group_time = time[p,:,s]
        group_normalization = norm[p,:,:]
        group_seg = data_seg[p, :, :, :, s, np.newaxis]

        yield (group_images, group_time, group_normalization, group_seg, s)
