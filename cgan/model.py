from cgan import preprocessing, util
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
import time
import tifffile
import torch
import torch.nn as nn

def conditional_trainer(pth, imtype, datatype, real_data, labels, Disc, Gen, isotropic, nc, l, nz, sf):
    print('Loading Dataset...')
    datasetxyz = preprocessing.cbatch(real_data,labels, datatype, l, sf,TI=True)
    ## Constants for NNs
    # matplotlib.use('Agg')
    ngpu = 1
    nlabels = len(labels[0])
    batch_size = 32
    num_epochs = 30
    lrg = 0.0002
    lr = 0.0001
    beta1 = 0
    beta2 = 0.9
    Lambda = 10
    critic_iters = 5
    cudnn.benchmark = True
    workers = 0

    ##Dataloaders for each orientation
    device = torch.device("cuda:0" if(torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(device, " will be used.\n")
    # device = torch.device('cpu')

    dataloader = torch.utils.data.DataLoader(datasetxyz, batch_size=batch_size,
                                              shuffle=True, num_workers=workers)

    # Create the Genetator network
    netG = Gen().to(device)
    if ('cuda' in str(device)) and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
    optG = optim.Adam(netG.parameters(), lr=lrg, betas=(beta1, beta2))

    # Define 1 discriminator and optimizer for each plane in each dimension

    netD = Disc()
    netD = (nn.DataParallel(netD, list(range(ngpu)))).to(device)
    optD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))

    disc_real_log = []
    disc_fake_log = []
    gp_log = []
    Wass_log = []

    print("Starting Training Loop...")
    # For each epoch
    start = time.time()
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, (data,lbl) in enumerate(dataloader):
            netD.zero_grad()
            G_labels = lbl.repeat(1, 1, 4, 4, 4).to(device)
            D_labels = lbl.repeat(1, 1, l, l, l)
            ### Discriminator
            ## Generate fake image batch with G
            noise = torch.randn(batch_size, nz, 4, 4, 4, device=device)
            fake_data = netG(noise, G_labels).detach()
            out_fake = netD(fake_data, D_labels).mean()

            real_data = data.to(device)
            out_real = netD(real_data, D_labels).view(-1).mean()
            gradient_penalty = util.cond_calc_gradient_penalty(netD, real_data, fake_data,batch_size,
                                                               l,device, Lambda, nc, D_labels)
            disc_cost = out_fake - out_real + gradient_penalty
            disc_cost.backward()
            optD.step()

            disc_real_log.append(out_real.item())
            disc_fake_log.append(out_fake.item())
            Wass_log.append(out_real.item() - out_fake.item())
            gp_log.append(gradient_penalty.item())
            ### Generator Training
            if i % int(critic_iters) == 0:
                netG.zero_grad()
                noise = torch.randn(batch_size, nz, 4, 4, 4, device=device)
                fake_data = netG(noise, G_labels)
                output = netD(fake_data, D_labels)
                errG = -output.mean()
                # Calculate gradients for G
                errG.backward()
                optG.step()
                # Output training stats & show imgs
            if i % 25 == 0:
                start_save = time.time()
                torch.save(netG.state_dict(), pth + '_Gen.pt')
                torch.save(netD.state_dict(), pth + '_Disc.pt')
                noise = torch.randn(1, nz, 4, 4, 4, device=device)
                for tst_lbls in labels:
                    lbl = torch.zeros(1, nlabels * 2, 4, 4, 4)
                    lbl_str = ''
                    for lb in range(nlabels):
                        lbl[:, lb] = tst_lbls[lb]
                        lbl[:, lb + nlabels] = 1- tst_lbls[lb]

                        lbl_str += str(tst_lbls[lb])
                    img = netG(noise, lbl.type(torch.FloatTensor).cuda())

                    util.test_plotter(img, 3, imtype, pth+lbl_str)

                ###Print progress
                ## calc ETA
                steps = len(dataloader)
                util.calc_eta(steps, time.time(), start, i, epoch, num_epochs)
                ###save example slices
                # plotting graphs
                util.graph_plot([disc_real_log, disc_fake_log], ['real', 'perp'], pth, 'LossGraph')
                util.graph_plot([Wass_log], ['Wass Distance'], pth, 'WassGraph')
                util.graph_plot([gp_log], ['Gradient Penalty'], pth, 'GpGraph')
                fin_save = time.time() - start_save
                # print('save: ', fin_save)
