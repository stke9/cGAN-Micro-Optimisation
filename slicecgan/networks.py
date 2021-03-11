import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle


def slicecgan_nets(pth, Training, lbls, *args):
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
    class GeneratorWGAN(nn.Module):
        def __init__(self):
            super(GeneratorWGAN, self).__init__()
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            for lay, (k,s,p) in enumerate(zip(gk,gs,gp)):
                self.convs.append(nn.ConvTranspose3d(gf[lay] if lay != 0 else gf[lay]+lbls, gf[lay+1], k, s, p, bias=False))
                self.bns.append(nn.BatchNorm3d(gf[lay+1]))

        def forward(self, x, y):
            x = torch.cat([x, y], 1)
            for lay, (conv,bn) in enumerate(zip(self.convs[:-1],self.bns[:-1])):
                x = F.relu_(bn(conv(x)))
            # out = torch.sigmoid(self.convs[-1](x))
            out = torch.softmax(self.convs[-1](x),1)
            return out

    class DiscriminatorWGAN(nn.Module):
        def __init__(self):
            super(DiscriminatorWGAN, self).__init__()
            self.convs = nn.ModuleList()
            for lay, (k, s, p) in enumerate(zip(dk, ds, dp)):
                self.convs.append(nn.Conv2d(df[lay] if lay != 0 else df[lay]+lbls, df[lay + 1], k, s, p, bias=False))

        def forward(self, x, y):
            x = torch.cat([x, y], 1)
            for lay, conv in enumerate(self.convs[:-1]):
                x = F.relu_(conv(x))
            x = self.convs[-1](x)
            return x
    print('Architect Complete...')


    return DiscriminatorWGAN, GeneratorWGAN

def slicecgan_rc_nets(pth, Training, lbls, *args):
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
    class GeneratorWGAN(nn.Module):
        def __init__(self):
            super(GeneratorWGAN, self).__init__()
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            self.rcconv = nn.Conv3d(gf[-2],gf[-1],3,1,0)
            for lay, (k,s,p) in enumerate(zip(gk,gs,gp)):
                self.convs.append(nn.ConvTranspose3d(gf[lay] if lay != 0 else gf[lay]+lbls, gf[lay+1], k, s, p, bias=False))
                self.bns.append(nn.BatchNorm3d(gf[lay+1]))

        def forward(self, x, y):
            x = torch.cat([x, y], 1)
            for lay, (conv, bn) in enumerate(zip(self.convs[:-1],self.bns[:-1])):
                x = F.relu_(bn(conv(x)))
            up = nn.Upsample(size = x.shape[2]*2-2)
            out = torch.softmax(self.rcconv(up(x)), 1)
            return out

    class DiscriminatorWGAN(nn.Module):
        def __init__(self):
            super(DiscriminatorWGAN, self).__init__()
            self.convs = nn.ModuleList()
            for lay, (k, s, p) in enumerate(zip(dk, ds, dp)):
                self.convs.append(nn.Conv2d(df[lay] if lay != 0 else df[lay]+lbls, df[lay + 1], k, s, p, bias=False))

        def forward(self, x, y):
            x = torch.cat([x, y], 1)
            for lay, conv in enumerate(self.convs[:-1]):
                x = F.relu_(conv(x))
            x = self.convs[-1](x)
            return x
    print('Architect Complete...')


    return DiscriminatorWGAN, GeneratorWGAN

def slicecgan_rc_pc_nets(pth, Training, lbls, *args):
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
    class GeneratorWGAN(nn.Module):
        def __init__(self):
            super(GeneratorWGAN, self).__init__()
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            self.rcconv = nn.Conv3d(gf[-2],gf[-1],3,1,0)
            self.pcconv = nn.Conv3d(gf[0],gf[0],3,1,0)


            for lay, (k,s,p) in enumerate(zip(gk,gs,gp)):
                self.convs.append(nn.ConvTranspose3d(gf[lay] if lay != 0 else gf[lay]+lbls, gf[lay+1], k, s, p, bias=False))
                self.bns.append(nn.BatchNorm3d(gf[lay+1]))

        def forward(self, x, y):
            x = self.pcconv(x)
            x = torch.cat([x, y[:,:,:4,:4,:4]], 1)
            for lay, (conv, bn) in enumerate(zip(self.convs[:-1],self.bns[:-1])):
                x = F.relu_(bn(conv(x)))
            up = nn.Upsample(size = x.shape[2]*2-2)
            out = torch.softmax(self.rcconv(up(x)), 1)
            return out

    class DiscriminatorWGAN(nn.Module):
        def __init__(self):
            super(DiscriminatorWGAN, self).__init__()
            self.convs = nn.ModuleList()
            for lay, (k, s, p) in enumerate(zip(dk, ds, dp)):
                self.convs.append(nn.Conv2d(df[lay] if lay != 0 else df[lay]+lbls, df[lay + 1], k, s, p, bias=False))

        def forward(self, x, y):
            x = torch.cat([x, y], 1)
            for lay, conv in enumerate(self.convs[:-1]):
                x = F.relu_(conv(x))
            x = self.convs[-1](x)
            return x
    print('Architect Complete...')


    return DiscriminatorWGAN, GeneratorWGAN


def slicecgan_resnets(pth, Training, lbls, *args):
    ##
    # save params
    params = [*args]

    if Training:
        dk, ds, df, dp, gk, gs, gf, gp = params
        with open(pth + '_params.data', 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(params, filehandle)
    else:
        with open(pth + '_params.data', 'rb') as filehandle:
            # read the data as binary data stream
            dk, ds, df, dp, gk, gs, gf, gp = pickle.load(filehandle)

    # else:
    #     with open(pth + Proj + '/' + Proj + '_parameters.txt', 'w') as f:
    # Make nets
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            lblf = 16
            self.lblconv = nn.ConvTranspose3d(lbls, lblf, gk[0], gs[0], gp[0], bias=False)
            self.lblbn = nn.BatchNorm3d(lblf)
            self.preconv = nn.Conv3d(gf[0], gf[0], 3, 1, 0)
            self.preconvbn = nn.BatchNorm3d(gf[0])
            self.postconv = nn.Conv3d(gf[-2], gf[-1], 3, 1, 0)
            for lay, (k, s, p) in enumerate(zip(gk, gs, gp)):
                self.convs.append(
                    nn.ConvTranspose3d(gf[lay] if lay != 1 else gf[lay] + lblf, gf[lay + 1], k, s, p, bias=False))
                self.bns.append(nn.BatchNorm3d(gf[lay + 1]))

        def forward(self, x, y):
            activations = []
            for lay, (conv, bn) in enumerate(zip(self.convs[:-1], self.bns[:-1])):
                if lay == 0:
                    x = F.relu_(bn(conv(self.preconvbn(self.preconv(x)))))
                    y = F.relu_(self.lblbn(self.lblconv(y)))
                    x = torch.cat([x, y], 1)
                    # activations.append(x)
                else:
                    new_size = (x.shape[2] - 1) * 2
                    up = nn.Upsample(size=new_size, mode='trilinear', align_corners=False)
                    x_res = up(x)
                    oc = conv.out_channels
                    x = F.relu_(bn(conv(x) + x_res[:, :oc]))
                    # activations.append(x)
            up = nn.Upsample(size=2 * x.size()[2] - 2, mode='trilinear', align_corners=False)
            out = torch.softmax(self.postconv(up(x)), 1)
            # activations.append(out)
            return out

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.convs = nn.ModuleList()
            lblf = 128
            self.lblconv = nn.Conv2d(lbls, lblf, dk[0], ds[0], dp[0], bias=False)
            for lay, (k, s, p) in enumerate(zip(dk, ds, dp)):
                self.convs.append(
                    nn.Conv2d(df[lay] if lay != 1 else df[lay] + lblf, df[lay + 1], k, s, p, bias=False))

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

def slicecgan_ps_resnets(pth, Training, lbls, *args):
    ##
    # save params
    params = [*args]

    if Training:
        dk, ds, df, dp, gk, gs, gf, gp = params
        with open(pth + '_params.data', 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(params, filehandle)
    else:
        with open(pth + '_params.data', 'rb') as filehandle:
            # read the data as binary data stream
            dk, ds, df, dp, gk, gs, gf, gp = pickle.load(filehandle)

    # else:
    #     with open(pth + Proj + '/' + Proj + '_parameters.txt', 'w') as f:
    # Make nets
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            lblf = 16
            self.lblconv = nn.ConvTranspose3d(lbls, lblf, gk[0], gs[0], gp[0], bias=False)
            self.lblbn = nn.BatchNorm3d(lblf)
            self.preconv = nn.Conv3d(gf[0], gf[0], 3, 1, 0)
            self.ps = PixelShuffle3d(2, 2)
            for lay, (k, s, p) in enumerate(zip(gk, gs, gp)):
                self.convs.append(
                    nn.ConvTranspose3d(gf[lay] if lay != 1 else gf[lay] + lblf, gf[lay + 1], k, s, p, bias=False))
                self.bns.append(nn.BatchNorm3d(gf[lay + 1]))

        def forward(self, x, y):
            activations = []
            for lay, (conv, bn) in enumerate(zip(self.convs[:-1], self.bns[:-1])):
                if lay == 0:
                    x = F.relu_(bn(conv(self.preconv(x))))
                    y = F.relu_(self.lblbn(self.lblconv(y)))
                    x = torch.cat([x, y], 1)
                    # activations.append(x)
                else:
                    new_size = (x.shape[2] - 1) * 2
                    up = nn.Upsample(size=new_size, mode='trilinear', align_corners=False)
                    x_res = up(x)
                    oc = conv.out_channels
                    x = F.relu_(bn(conv(x) + x_res[:, :oc]))
                    # activations.append(x)
            out = torch.softmax(self.ps(x), 1)
            # activations.append(out)
            return out

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.convs = nn.ModuleList()
            lblf = 128
            self.lblconv = nn.Conv2d(lbls, lblf, dk[0], ds[0], dp[0], bias=False)
            for lay, (k, s, p) in enumerate(zip(dk, ds, dp)):
                self.convs.append(
                    nn.Conv2d(df[lay] if lay != 1 else df[lay] + lblf, df[lay + 1], k, s, p, bias=False))

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

class PixelShuffle3d(nn.Module):
    '''
    This class is a 3d version of pixelshuffle.
    '''
    def __init__(self, scale, p = 0):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale = scale
        self.p = p

    def forward(self, input):
        p = self.p
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale ** 3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        output = output.view(batch_size, nOut, out_depth, out_height, out_width)
        if p:
            return output[:, :, p:-p, p:-p, p:-p]
        return output