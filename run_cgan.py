### Welcome to SliceGAN ###
####### Steve Kench #######

from cgan import *

## Make directory

Project_name = 'NMC_Alej_3D_restc'
Project_dir = 'trained_generators/NMC_Alej/'

## Data Processing
image_type = 'threephase' # threephase, twophase or colour
data_type = 'self' # png, jpg, tif, array, array2D
data_path = []
labels = []
# # # Scotts labels

# for ca, ca_lab in zip(['000.10','100.00'], [0, 1]):
#     for cc, cc_lab  in zip(['000.10','100.00'], [0, 1]):
#         for por, por_lab in zip(['30','40','50'], [0, 0.5, 1]):
#             # for wt_lab, wt in enumerate(zip(['90','96']),1):
#             file = 'ds_wt0.92_ca{}_cc{}_case01_porosity0.{}_phases.npy'.format(ca, cc, por)
#             data_path.append('Examples/Scott_NMC/round1_2/'+ file) # path to training data.
#             labels.append([ca_lab, cc_lab, por_lab])
# Alej labels
for lab, NMC in zip(['94','95', '96'],[0, 0.5, 1]):
    data_path.append('training_data/Alej_NMC/batch2/{}.npy'.format(lab))
    labels.append([NMC])

isotropic = True
Training = False # Run with False to show an image during training
Project_path = mkdr(Project_name, Project_dir, Training)

# Network Architectures
imsize, nz,  channels, sf, lbls = 64, 8, 3, 2, len(labels[0]*2)
lays = 5
laysd = 5

dk, gk = [4]*lays, [4]*lays
#gk[3] = 4
ds, gs = [2]*lays, [2]*lays
df, gf = [channels,64,128,256,512,1], [nz,512,256,128, 64, channels]
dp, gp = [1,1,1,1,0],[2,2,2,2,3]
#gp = [0,0,0,0,0]

# dk, gk = [4]*lays, [4]*lays
# ds, gs = [2]*lays, [2]*lays
# df, gf = [channels,64,128,256,384,1], [nz,384,256,128, 64, channels]
# dp, gp = [1,1,1,1,0],[2,2,2,2,3]

##Create Networks
netD, netG = cgan_resnets(Project_path, Training, lbls, dk, ds, df,dp, gk ,gs, gf, gp)

if Training:
    data = conditional_trainer(Project_path, image_type, data_type, data_path, labels, netD, netG, isotropic, channels, imsize, nz, sf)

else:
    img, raw, netG = test_img_cgan(Project_path, labels, image_type, netG(), nz, show=False, lf=6)
    for im in img:
        for ph in [0, 1, 2]:
            print(len(im[im == ph]) / im.size)

