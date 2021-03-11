from slicecgan.util import *
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
import time
import tifffile
import torch
import torch.nn as nn

def conditional_trainer(pth, imtype, datatype, real_data, labels, Disc, Gen, isotropic, nc, l, nz, sf):
    print('Loading Dataset...')
    ## Constants for NNs
    # matplotlib.use('Agg')
    ngpu = 1
    nlabels = len(labels[0])
    batch_size = 27
    D_batch_size = 27
    num_epochs = 100
    iters = 30000//batch_size
    lrg = 0.0004
    lr = 0.0002
    beta1 = 0.9
    beta2 = 0.999
    Lambda = 10
    critic_iters = 5
    cudnn.benchmark = True
    workers = 0
    lz = 6
    ##Dataloaders for each orientation
    device = torch.device("cuda:0" if(torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(device, " will be used.\n")

    training_imgs = pre_proc(real_data, sf)

    # Create the Genetator network
    netG = Gen().to(device)
    if ('cuda' in str(device)) and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
    optG = optim.Adam(netG.parameters(), lr=lrg, betas=(beta1, beta2))

    # Define 1 discriminator and optimizer for each plane in each dimension
    netDs = []
    optDs = []
    for i in range(3):
        netD = Disc()
        netD = (nn.DataParallel(netD, list(range(ngpu)))).to(device)
        netDs.append(netD)
        optDs.append(optim.Adam(netDs[i].parameters(), lr=lr, betas=(beta1, beta2)))

    disc_real_log = []
    disc_fake_log = []
    gp_log = []
    Wass_log = []

    print("Starting Training Loop...")
    # For each epoch
    start = time.time()
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i in range(iters):
            real_data, lbl = batch(training_imgs, labels, l, D_batch_size, device)
            G_labels = lbl.repeat(1, 1, lz, lz, lz).to(device)
            D_labels_real = lbl.repeat(1, 1, l, l, l)[:, :, 0]
            D_labels_fake = D_labels_real.repeat(1, l, 1 ,1).reshape(-1, nlabels*2, l, l)
            ### Discriminator
            ## Generate fake image batch with G
            noise = torch.randn(D_batch_size, nz, lz, lz, lz, device=device)
            fake_data = netG(noise, G_labels).detach()
            # For each dimension
            start_disc = time.time()
            for dim, (netD, optimizer, d1, d2, d3) in enumerate(
                    zip(netDs, optDs, [2, 3, 4], [3, 2, 2], [4, 4, 3])):
                if isotropic:
                    netD = netDs[0]
                    optimizer = optDs[0]
                ##train on real images
                netD.zero_grad()
                # Forward pass real batch through D
                out_real = netD(real_data, D_labels_real).view(-1).mean()
                # train on fake images
                fake_data_perm = fake_data.permute(0, d1, 1, d2, d3).reshape(l * D_batch_size, nc, l, l)
                out_fake = netD(fake_data_perm, D_labels_fake).mean()
                #grad calc
                gradient_penalty = cond_calc_gradient_penalty(netD, real_data, fake_data_perm[::l],
                                                              D_batch_size, l,
                                                              device, Lambda, nc, D_labels_real)

                disc_cost = out_fake - out_real + gradient_penalty
                disc_cost.backward()
                optimizer.step()

            disc_real_log.append(out_real.item())
            disc_fake_log.append(out_fake.item())
            Wass_log.append(out_real.item() - out_fake.item())
            gp_log.append(gradient_penalty.item())
            ### Generator Training
            start_gen = time.time()
            if i % int(critic_iters) == 0:
                netG.zero_grad()
                errG = 0
                noise = torch.randn(D_batch_size, nz, lz, lz, lz, device=device)
                fake = netG(noise, G_labels)
                for dim, (netD, d1, d2, d3) in enumerate(
                        zip(netDs, [2, 3, 4], [3, 2, 2], [4, 4, 3])):
                    if isotropic:
                        netD = netDs[0]
                    # For each plane
                    fake_data_perm = fake.permute(0, d1, 1, d2, d3).reshape(l * D_batch_size, nc, l, l)
                    output = netD(fake_data_perm, D_labels_fake)
                    errG -= output.mean()
                    # Calculate gradients for G
                errG.backward()
                optG.step()
            # Output training stats & show imgs
            if i % 25 == 0:
                start_save = time.time()
                torch.save(netG.state_dict(), pth + '_Gen.pt')
                torch.save(netD.state_dict(), pth + '_Disc.pt')
                noise = torch.randn(1, nz, lz, lz, lz, device=device)
                for tst_lbls in labels[::9]:
                    lbl = torch.zeros(1, nlabels * 2, lz, lz, lz)
                    lbl_str = ''
                    for lb in range(nlabels):
                        lbl[:, lb] = tst_lbls[lb]
                        lbl[:, lb + nlabels] = 1- tst_lbls[lb]

                        lbl_str += '_' + str(tst_lbls[lb])
                    img = netG(noise, lbl.type(torch.FloatTensor).cuda())

                    test_plotter(img, 3, imtype, pth+lbl_str)

                ###Print progress
                ## calc ETA
                calc_eta(iters, time.time(), start, i, epoch, num_epochs)
                ###save example slices
                # plotting graphs
                graph_plot([disc_real_log, disc_fake_log], ['real', 'fake'], pth, 'LossGraph')
                graph_plot([Wass_log], ['Wass Distance'], pth, 'WassGraph')
                graph_plot([gp_log], ['Gradient Penalty'], pth, 'GpGraph')
                fin_save = time.time() - start_save
                # print('save: ', fin_save)
