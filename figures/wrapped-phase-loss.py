#!/usr/bin/env python

import matplotlib
import matplotlib.pylab as plt

plt.rc('text', usetex=True)
#font = {'family':'serif','size':16}
font = {'family':'serif','size':14, 'serif': ['computer modern roman']}
plt.rc('font',**font)
plt.rc('legend',**{'fontsize':12})
matplotlib.rcParams['text.latex.preamble']=[r'\usepackage{amsmath}']

import numpy as np
import pylab as plt
import os
import dvpy as dv

from matplotlib import cm 
from mpl_toolkits.mplot3d import Axes3D

s = 200
g = np.linspace(-np.pi, np.pi, s)
p = np.linspace(-np.pi, np.pi, s)

G, P = np.meshgrid(g, p)

##
##
##

fig1 = plt.figure(figsize=(6, 5))
fig2 = plt.figure(figsize=(6, 5))
ax1 = fig1.add_subplot(1, 1, 1, projection = '3d')
ax2 = fig2.add_subplot(1, 1, 1, projection = '3d')

ax1.plot_surface(G, P, (P - G)**2, cmap=cm.jet, rcount = s, ccount = s)
ax2.plot_surface(G, P, (dv.wrap_phase(P - G))**2, cmap=cm.jet, rcount = s, ccount = s)

for ax in [ax1, ax2]:
  ax.set_xlabel(r'$\hat{\theta}$', labelpad = 10)
  ax.set_ylabel(r'$\theta$'      , labelpad = 10)
  ax.invert_yaxis()
  ax.view_init(elev = 20.0, azim = -80.0)
  ax.set_zlim([0,45])
  ax.set_xlim([-np.pi, +np.pi])
  ax.set_ylim([-np.pi, +np.pi])
  ax.set_xticks([-np.pi, 0.0, np.pi])
  ax.set_yticks([-np.pi, 0.0, np.pi])
  ax.set_xticklabels(['$-\pi$', '$0$', '$+\pi$'])
  ax.set_yticklabels(['$-\pi$', '$0$', '$+\pi$'])

ax1.set_zlabel(r'$\frac{1}{2} (\theta - \hat{\theta})^2$'           , labelpad = 10)
ax2.set_zlabel(r'$\frac{1}{2} \left(\mathcal{W} ( \theta - \hat{\theta} ) \right)^2$', labelpad = 10)

#plt.tight_layout()
#plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.0, hspace=0.0)

fig1.savefig(os.path.expanduser('~/Dropbox/Cardiac_Segmentation/manuscript-ohm-net-data/phase-loss.png'), transparent = True)
fig2.savefig(os.path.expanduser('~/Dropbox/Cardiac_Segmentation/manuscript-ohm-net-data/wrapped-phase-loss.png'), transparent = True)


