import numpy as np

def volfrac(img, phases='all'):
    """
    calculate phase volume fractions of an image
    :param img: n phase input
    :param phases: the phases to be calculated, default all
    :return: list of VFs
    """
    if phases == 'all':
        phases = np.unique(img)
    if isinstance(phases, int):
        print('listifying')
        phases = [phases]

    vfs = []
    imsize = img.size
    for ph in phases:
        vfs.append(len(img[img == ph])/imsize)
    if len(vfs) == 1:
        return vfs[0]
    return vfs

def surface_area(img, ph1, ph2):
    """
    calcualte interfactial surface area between two phases in a volume
    :param img:
    :param ph1: first phase to calculate SA
    :param ph2: second phase to calculate SA.
    :return: the surface area in faces per unit volume
    """

    img = np.pad(img, 1, 'constant', constant_values=-1)
    SA_map = np.zeros_like(img)
    # create phase map for ph1

    for dr in range(len(img.shape)):
        for sh in [1, -1]:
            #shift img by one
            new_img = np.roll(img, sh, dr)
            #create phase map for ph2
            SA_map[(new_img == ph2) & (img == ph1)] += 1
    # remove padding
    SA_map = SA_map[1:-1, 1:-1]
    if len(img.shape) == 3:
        SA_map = SA_map[:, :, 1:-1]
    return np.mean(SA_map)


