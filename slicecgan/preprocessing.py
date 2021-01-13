import numpy as np
import torch
import matplotlib.pyplot as plt
import tifffile

def cbatch(imgs,lbls, typ, l, sf, TI):
    if typ == 'self':
        nlabs = len(lbls[0])
        data = np.empty([32 * 900, 3, l, l])
        labelset = np.empty([32 * 900, nlabs * 2, 1, 1, 1])
        p = 0
        nimgs = len(imgs)
        print('number of training imgs: ', nimgs, ' number of labels: ', nlabs)
        for imgpth,lbl in zip(imgs,lbls):
            img = np.load(imgpth)
            if len(img.shape) > 3:
                img = img[:, :, 0]
            img = img[::sf, ::sf]
            x_max, y_max, z_max = img.shape[:]
            phases = np.unique(img)
            for i in range((32//nimgs) * 900):
                for j,lb in enumerate(lbl):
                    labelset[p, j] = lb
                    labelset[p, j+nlabs] = 1 -lb
                    if i == 0: print(str(lb) +'\n' + str(lbl) + '\n' + imgpth)
                x = np.random.randint(1, x_max - l - 1)
                y = np.random.randint(1, y_max - l - 1)
                z = np.random.randint(1,z_max)
                # create one channel per phase for one hot encoding
                for cnt, phs in enumerate(phases):
                     img1 = np.zeros([l, l])
                     img1[img[x:x + l, y:y + l, z] == phs] = 1
                     data[p, cnt, :, :] = img1
                p+=1
                if i%5000==0:
                    plt.imshow(data[p-1,0] + 2*data[p-1,1])
                    plt.pause(1)
                    plt.close('all')
        data = torch.FloatTensor(data)
        labelset = torch.FloatTensor(labelset)
        dataset = torch.utils.data.TensorDataset(data,labelset)
        return dataset
