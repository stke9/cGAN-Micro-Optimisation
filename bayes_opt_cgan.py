from bayes_opt import BayesianOptimization as BO
from slicecgan import *
import numpy as np
from taufactor.taunet_cupy import TauNetCupy as TauNet
# Preloading

Project_name = 'NMC_Alej_2D'
Project_dir = 'trained_generators/NMC_Alej/'

## Data Processing
image_type = 'threephase' # threephase, twophase or colour

isotropic = False
Training = False # Run with False to show an image during training
Project_path = mkdr(Project_name, Project_dir, Training)

# Network Architecture
imsize, nz,  channels, sf, lbls = 64, 8, 3, 2, 2
nc = 3
##Create Networks
netD, netG = slicecgan_nets(Project_path, Training, lbls)

netG = netG()
netG.eval()
netG.load_state_dict(torch.load(Project_path + '_Gen.pt'))
netG.cuda()
# netD = netD()
# netD = nn.DataParallel(netD)
# netD.load_state_dict(torch.load(Project_path + '_Disc.pt'))
# netD.eval()
# netD.cuda()

# optimisation params
lf = 8
batch_size = 1
noise = torch.randn(batch_size, nz, lf, lf, lf, requires_grad = True).cuda()


def generator(lbl):
    """
    generators image given the current label
    :param lbl: label for G
    :return: OHE image
    """
    labels = torch.zeros((batch_size, 2, lf, lf, lf)).cuda()
    labels[:,0] = lbl
    labels[:,1] = 1-lbl
    out = post_proc(netG(noise, labels), image_type)
    return out

def opt_func(lbl):
    """
    calculates current loss
    :param lbl: label for G
    :return: the value of the function to be optimised at position 'lbl'
    """
    img = generator(lbl)
    Deff, porosity = TauNet(img).solve(9)
    print(Deff, porosity)
    return Deff / porosity

# label bounds
opts = []
p_bounds = {'lbl': (0, 1)}
for min_vf in np.arange(0.38, 0.45,0.01):
    optimizer = BO(
        f=opt_func,
        pbounds=p_bounds,
        verbose=2,
        random_state=1
    )

    optimizer.maximize(
        init_points=4,
        n_iter=20,
        kappa=8,

    )
    opts.append(optimizer)

cmap = plt.cm.get_cmap('viridis', len(opts))
for i, opt in enumerate(opts):
    max_res = 0
    for res in opt.res:
        if res['target'] > max_res:
            max_res = res['target']
    for res in opt.res:
        plt.scatter(res['params']['lbl'], res['target']/max_res, color=cmap(i))