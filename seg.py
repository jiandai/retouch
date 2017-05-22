# ver 20170519 by jian: per-pixel one-hot coding

import SimpleITK as sitk
ref=sitk.ReadImage('data/Cirrus_part1/3c68f67cd2e2b41afa54bf6059f509d1/reference.mhd')
ref_arr = sitk.GetArrayFromImage(ref)
print(ref_arr.shape)
quit()
import keras
# One-hot encoding the mask
ref_1h = keras.utils.to_categorical(ref_arr,num_classes=4)
ref_1h_arr = ref_1h.reshape(128, 1024, 512,4)

'''
# Compare and validate:
one_h = np.zeros([128,1024,512,4])
for i in range(128):
    for j in range(1024):
        for k in range(512):
            one_h[i,j,k,ref_arr[i,j,k]]=1


(one_h != ref_1h_arr).sum() # 0
'''


import numpy as np


