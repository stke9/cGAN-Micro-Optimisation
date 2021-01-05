import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

def cgan_nets(pth, Training, lbls, *args):
    ##
    #save params
    params = [*args]

    if Training:
        dk, ds, df, dp, gk, gs, gf, gp = params
        with open(pth + '_params.data', 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(params, filehandle)
    else:
        with open(pth + '_params.data', 'rb') as filehandle:
            # read the data as binary data stream
            dk, ds, df, dp, gk, gs, gf, gp  = pickle.load(filehandle)
    # else:
    #     with open(pth + Proj + '/' + Proj + '_parameters.txt', 'w') as f:
    #Make nets
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            lblf = 16
            self.lblconv = nn.ConvTranspose3d(lbls, lblf, gk[0], gs[0], gp[0], bias=False)
            self.lblbn = nn.BatchNorm3d(lblf)
            for lay, (k,s,p) in enumerate(zip(gk,gs,gp)):
                self.convs.append(nn.ConvTranspose3d(gf[lay] if lay != 1 else gf[lay]+lblf, gf[lay+1], k, s, p, bias=False))
                self.bns.append(nn.BatchNorm3d(gf[lay+1]))

        def forward(self, x, y):
            for lay, (conv,bn) in enumerate(zip(self.convs[:-1],self.bns[:-1])):
                x = F.relu_(bn(conv(x)))
                if lay == 0:
                    y = F.relu_(self.lblbn(self.lblconv(y)))
                    x = torch.cat([x, y], 1)
            out = torch.softmax(self.convs[-1](x),1)
            return out

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.convs = nn.ModuleList()
            lblf = 128
            self.lblconv = nn.Conv3d(lbls, lblf, dk[0], ds[0], dp[0], bias=False)
            for lay, (k, s, p) in enumerate(zip(dk, ds, dp)):
                self.convs.append(nn.Conv3d(df[lay] if lay != 1 else df[lay]+lblf, df[lay + 1], k, s, p, bias=False))

        def forward(self, x, y):
            for lay, conv in enumerate(self.convs[:-1]):
                x = F.relu_(conv(x))
                if lay == 0:
                    y = F.relu_(self.lblconv(y))
                    x = torch.cat([x, y], 1)
            x = self.convs[-1](x)
            return x
    print('Architect Complete...')


    return Discriminator, Generator

def cgan_resize_conv_nets(pth, Training, lbls, *args):
    ##
    #save params
    params = [*args]

    if Training:
        dk, ds, df, dp, gk, gs, gf, gp = params
        with open(pth + '_params.data', 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(params, filehandle)
    else:
        with open(pth + '_params.data', 'rb') as filehandle:
            # read the data as binary data stream
            dk, ds, df, dp, gk, gs, gf, gp  = pickle.load(filehandle)
    # else:
    #     with open(pth + Proj + '/' + Proj + '_parameters.txt', 'w') as f:
    #Make nets
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            lblf = 16
            self.lblconv = nn.Conv3d(lbls, lblf, gk[0], gs[0], gp[0], bias=False)
            self.lblbn = nn.BatchNorm3d(lblf)
            for lay, (k,s,p) in enumerate(zip(gk,gs,gp)):
                self.convs.append(nn.Conv3d(gf[lay] if lay != 1 else gf[lay]+lblf, gf[lay+1], k, s, p, bias=False))
                self.bns.append(nn.BatchNorm3d(gf[lay+1]))

        def forward(self, x, y):
            upS = nn.Upsample(scale_factor=2)
            for lay, (conv,bn) in enumerate(zip(self.convs[:-1],self.bns[:-1])):
                x = F.relu_(bn(conv(upS(x))))
                if lay == 0:
                    y = F.relu_(self.lblbn(self.lblconv(upS(y))))
                    x = torch.cat([x, y], 1)
            out = torch.softmax(self.convs[-1](upS(x)),1)
            return out

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.convs = nn.ModuleList()
            lblf = 4
            self.lblconv = nn.Conv3d(lbls, lblf, dk[0], ds[0], dp[0], bias=False)
            for lay, (k, s, p) in enumerate(zip(dk, ds, dp)):
                self.convs.append(nn.Conv3d(df[lay] if lay != 1 else df[lay]+lblf, df[lay + 1], k, s, p, bias=False))

        def forward(self, x, y):
            for lay, conv in enumerate(self.convs[:-1]):
                x = F.relu_(conv(x))
                if lay == 0:
                    y = F.relu_(self.lblconv(y))
                    x = torch.cat([x, y], 1)
            x = self.convs[-1](x)
            return x
    print('Architect Complete...')


    return Discriminator, Generator

