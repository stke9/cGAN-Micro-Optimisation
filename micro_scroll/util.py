import slicecgan

def generate_image():

  Project_name = 'NMC_Alej_2D_fullres/'
  Project_dir = '../trained_generators/NMC_Alej/'

  ## Data Processing
  image_type = 'threephase'  # threephase, twophase or colour
  data_type = 'self'  # png, jpg, tif, array, array2D
  data_path = []
  labels = []
  # # # Scotts labels
  # Alej labels
  labels = [[0]]

  Training = 1  # Run with False to show an image during training
  Project_path = Project_dir + Project_name +Project_name[:-1]

  # Network Architectures
  imsize, nz, channels, sf, lbls = 64, 8, 3, 1, len(labels[0] * 2)
  lays = 5
  laysd = 5
  dk, gk = [4] * laysd, [4] * lays  # kernal sizes
  # gk[0]=8
  ds, gs = [2] * laysd, [2] * lays  # strides
  # gs[0] = 4
  df, gf = [channels, 64, 128, 256, 384, 1], [nz, 384, 256, 128, 64, channels]  # filter sizes for hidden layers
  dp, gp = [1, 1, 1, 1, 0], [2, 2, 2, 2, 3]

  ##Create Networks
  netD, netG = slicecgan.slicecgan_rc_nets(Project_path, Training, lbls, dk, ds, df, dp, gk, gs, gf, gp)
  img, raw, netG = slicecgan.test_img_cgan(Project_path, labels, image_type, netG(), nz, lf=8, twoph=0)
  return img[0]

def start_scroll():
  pass


def scroll():
  pass


def stop_scroll():
  pass


def update_screen():
  pass