from bayes_opt import BayesianOptimization as BO
from slicecgan import *
from taufactor import Solver as TauNet
from stat_analysis.metric_calcs import *
import pybamm
import sys, os
import pickle

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

Project_name = 'wt_psd_comp_rep0'
Project_dir = 'trained_generators/NMC_Alej/'

## Data Processing
image_type = 'threephase' # threephase, twophase or colour


isotropic = True
Training = 0 # Run with False to show an image during training
Project_path = mkdr(Project_name, Project_dir, Training)

##Create Networks
n_lbls = 3
netD, netG = slicecgan_rc_nets(Project_path, Training, n_lbls * 2)

netG = netG()
netG.load_state_dict(torch.load(Project_path + '_Gen.pt'))
netG.eval()

netG.cuda()


# optimisation params
lf = 10
nz = 32
batch_size = 1
noise = torch.randn(batch_size, nz, lf, lf, lf).cuda()


def generator(label):
    """
    generators image given the current label
    :param lbl: label for G
    :return: OHE image
    """
    y = torch.zeros((batch_size, n_lbls * 2, lf, lf, lf)).cuda()
    for ch, lbl in enumerate(label):
        y[:, ch] = lbl
        y[:, ch + n_lbls] = 1-lbl
    out = post_proc(netG(noise, y), image_type)
    return out

def opt_func(am_wt, psd, comp):
    """
    calculates current loss
    :param lbl: label for G
    :return: the value of the function to be optimised at position 'lbl'
    """
    label = [am_wt, psd, comp]
    with torch.no_grad():
        imgs = generator(label)
    Deff, am_wt, active_SA, porosity = 0, 0, 0, 0
    for l in range(batch_size):
        img = imgs[l]
        am_wt += volfrac(img, 0)
        porosity += volfrac(img, 1)
        active_SA += surface_area(img, 0, 1)
        img[img != 1] = 0
        blockPrint()
        Deff += TauNet(img).solve()
        enablePrint()
    return accessible_capacity(Deff, porosity, am_wt, current=current)

def accessible_capacity(neg_diff, neg_porosity, neg_am_frac, current=10):

    model = pybamm.lithium_ion.DFN()
    parameter_values = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Chen2020)
    parameter_values["Current function [A]"] = current
    parameter_values["Negative electrode diffusivity [m2.s-1]"] *= neg_diff
    parameter_values["Negative electrode porosity"] = neg_porosity
    parameter_values["Negative electrode active material volume fraction"] = neg_am_frac
    sim = pybamm.Simulation(model, parameter_values=parameter_values)
    sim.solve([0, 7200])
    return np.sum(sim.solution['Terminal voltage [V]'].entries) * current

# label bounds
opts = []
p_bounds = {'am_wt': (0, 1),
            'psd': (0, 1),
            'comp': (0, 1)}

for current in np.arange(4, 10, 0.3):
    print(current)
    optimizer = BO(
        f=opt_func,
        pbounds=p_bounds,
        verbose=2,
        random_state=1
    )

    optimizer.maximize(
        init_points=10,
        n_iter=50,
        acq='ei',
        xi=0.01
    )
    opts.append(optimizer)

cmap = plt.cm.get_cmap('viridis', len(opts))
fig, axs = plt.subplots(2, 4)
for i, (ax, opt) in enumerate(zip(axs.ravel(), opts)):
    targets = [res['target'] for res in opt.res]
    max_res = np.max(targets)
    min_res = np.min(targets)

    x = [res['params']['comp'] for res in opt.res]
    y = [res['params']['psd'] for res in opt.res]
    c = [res['target'] for res in opt.res]

    ax.scatter(x, y, c=c, cmap='viridis')
    max = np.argmax(c)
    # ax.scatter(x[max], y[max], color=(0, 0, 0), s=1.5)
    if i ==0:
        ax.set_xlabel('comp')
        ax.set_ylabel('psd')
    ax.set_title('current {}A'.format(i*1))
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=min_res, vmax=max_res))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.2)

# save
data = {}
for i, opt in enumerate(opts):
    data[str(i)] = opt.res

with open('optimisers1.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
