#!/usr/bin/env python

# System
import os
import glob as gb
import pathlib as plib

# Third Party
from PIL import Image
import numpy as np
import nibabel as nb
import pylab as plt
from skimage.exposure import equalize_hist
import scipy.ndimage.interpolation as itp

# Internal
import dvpy as dv

base = os.path.expanduser('~/Dropbox/oxford_data/simulation/')
out_base = os.path.expanduser('~/Dropbox/Cardiac_Segmentation/manuscript-ohm-net-data/ohm/')

patients = gb.glob(os.path.join(base, '*/HCMNet_*/*/*/'))

offsets = {'00_SAX' : np.pi,
           '01_HLA' : np.pi / 2.0,
           '02_VLA' : np.pi / 2.0,
          }

for p in patients:
  print(p)

  rotation = np.load(os.path.join(p, 'rotation.npy')) + offsets[dv.tokenize_path(p)[-2]]

  sg = np.asarray(nb.load(os.path.join(p, 'seg_gt.nii')).get_data())[:,:,0]
  sg = dv.crop_or_pad(sg, target = 256)
  xy, rt, zm, s_xy, s_rt, s_zm = dv.mask_to_transform(sg,
                                                      rotation = rotation,
                                                      margin = 0.1,
                                                      return_images = True)

  im = np.asarray(Image.open(os.path.join(p, 'ssfp/0.png')))
  im = dv.crop_or_pad(equalize_hist(dv.correct_nonuniform_illumination(im)), target = 256)
  i_xy = itp.shift(im, np.array(xy)*-128)
  i_rt = itp.rotate(i_xy, rotation*180/np.pi, reshape = False)
  i_zm = dv.crop_or_pad(itp.zoom(i_rt, 1.0/zm), target = 256)

  out_file = os.path.join(out_base, *dv.tokenize_path(p)[-4:])
  os.makedirs(out_file, exist_ok = True)

  dv.save_interpolated_image(im, os.path.join(out_file, 'im.png'))
  dv.save_interpolated_image(i_zm, os.path.join(out_file, 'im_t.png'))

