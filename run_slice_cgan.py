### Welcome to SliceGAN ###
####### Steve Kench #######

from slicecgan import *

## Make directory

Project_name = 'NMC_Alej_2D_fullres'
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
Training = 1 # Run with False to show an image during training
Project_path = mkdr(Project_name, Project_dir, Training)

# Network Architectures
imsize, nz,  channels, sf, lbls = 64, 8, 3, 1, len(labels[0]*2)
lays = 5
laysd = 5
dk, gk = [4]*laysd, [4]*lays                                    # kernal sizes
# gk[0]=8
ds, gs = [2]*laysd, [2]*lays                                    # strides
# gs[0] = 4
df, gf = [channels,64,128,256,384,1], [nz,384,256,128, 64, channels]  # filter sizes for hidden layers
dp, gp = [1, 1, 1, 1, 0], [2, 2, 2, 2, 3]

##Create Networks
netD, netG = slicecgan_rc_nets(Project_path, Training, lbls, dk, ds, df,dp, gk ,gs, gf, gp)

if Training:
    data = conditional_trainer(Project_path, image_type, data_type, data_path, labels, netD, netG, isotropic, channels, imsize, nz, sf)

else:
    img, raw, netG = test_img_cgan(Project_path, labels, image_type, netG(), nz,  lf=8, twoph=0)
    for im in img:
        for ph in [0, 1, 2]:
            print(len(im[im == ph]) / im.size)
    # netG = netG()
    # device = torch.device("cpu")
    # netG.load_state_dict(torch.load(Project_path + '_Gen.pt'))
    # netG.to(device)
    # labels = torch.ones([1, 2, 4, 4, 4])
    # labels[:, 0] = 0
    # out = netG(torch.randn(1, 128, 6, 6, 6), labels)
    # fig, axs = plt.subplots(7, 30)
    # for l, lay in enumerate(out):
    #     for p in range(30):
    #         try:
    #             axs[l, p].imshow(lay[0, p, 0].detach())
    #         except:
    #             pass
    # axs[-1, -1].imshow(post_proc(out[-1], image_type)[0])