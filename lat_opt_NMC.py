### Welcome to SliceGAN ###
####### Steve Kench #######

from slicecgan import *

Project_name = 'wt_psd_comp_rep0'
Project_dir = 'trained_generators/NMC_Alej/'

## Data Processing
image_type = 'threephase' # threephase, twophase or colour
data_type = 'self' # png, jpg, tif, array, array2D
data_path = []
labels = []
for wt, wt_lab in zip(['94', '95', '96'], [0, 0.5, 1]):
    for psd, psd_lab in zip(['0', '0.5', '1'], [0, 0.5, 1]):
        for comp, comp_lab in zip(['0','10','20'], [0, 0.5, 1]):
            # for wt_lab, wt in enumerate(zip(['90','96']),1):
            file = 'training_data/Alej_NMC/batch3/wt{}_psd1frac{}_comp{}.tif'.format(wt, psd, comp)
            data_path.append(file) # path to training data.
            labels.append([wt_lab, psd_lab, comp_lab])

isotropic = True
Training = False # Run with False to show an image during training
Project_path = mkdr(Project_name, Project_dir, Training)

# Network Architectures
imsize, nz,  channels, sf, lbls = 64, 32, 3, 2, 6
lays = 5
laysd = 5

dk, gk = [4]*lays, [4]*lays
ds, gs = [2]*lays, [2]*lays
df, gf = [channels,64,128,256,512,1], [nz,512,256,128, 64, channels]
dp, gp = [1,1,1,1,0],[2,2,2,2,3]

# dk, gk = [4]*lays, [3]*lays
# gk[3] = 4
# ds, gs = [2]*lays, [1]*lays
# df, gf = [channels,64,128,256,256,1], [nz,512,256,128, 64, channels]
# dp, gp = [1,1,1,1,0], [0,0,0,0,0]


##Create Networks
netD, netG = slicecgan_rc_nets(Project_path, Training, lbls, dk, ds, df,dp, gk ,gs, gf, gp)
netG = netG()
netG.eval()
netG.load_state_dict(torch.load(Project_path + '_Gen.pt'))
netG.cuda()

netD = netD()
netD = nn.DataParallel(netD)
netD.load_state_dict(torch.load(Project_path + '_Disc.pt'))
netD.eval()
netD.cuda()

lf = 8
iters = 9
l = 192
nc = 3
noise = torch.randn(1, nz, lf, lf, lf, requires_grad = True).cuda()
noise = torch.nn.Parameter(noise)

fake_labels = torch.zeros((1, lbls, 1, 1, 1)).cuda()
fake_labels[:, 3:] = 1

optimizer = torch.optim.Adam([noise], 0.01, (0.9,0.99))
imgs = []
criterion = nn.MSELoss()
for iter in range(iters):
    loss = 0
    img = netG(noise, fake_labels.repeat(1, 1, lf, lf, lf))
    # save init masses
    masses = [torch.mean(img[:, ch]) for ch in range(3)]
    if iter == 0:
        targ_masses = [m.detach() for m in masses]
    # loss to keep mass constant
    mass_losses = 0
    for targ, real in zip(targ_masses, masses):
        mass_losses += (targ - real)**2
    # if not iter:
    #     img0 = img.detach()[-1]
    # mass_losses = torch.mean((img-img0)**2)

    loss = 0
    for dim, (d1, d2, d3) in enumerate(zip([2, 3, 4], [3, 2, 2], [4, 4, 3])):
        fake_data_perm = img.permute(0, d1, 1, d2, d3).reshape(l, nc, l, l)
        loss -= torch.mean(netD(fake_data_perm, fake_labels[:,:,0].repeat(l,1,l,l)))
    # loss = criterion(loss/3, torch.tensor(-30.0).cuda())
    loss += mass_losses*1000
    loss.backward()
    print(loss.item(), mass_losses.item() * 100000, targ_masses, masses)
    with torch.no_grad():
        optimizer.step()
        noise.grad.zero_()
    imgs.append(post_proc(img, image_type)[0])


fig, axs = plt.subplots(3, 3)
for l in range(128):
    for p, ax in enumerate(axs.ravel()):
        ax.clear()
        ax.imshow(imgs[p][l])
    fig.canvas.draw()
    plt.pause(0.1)
