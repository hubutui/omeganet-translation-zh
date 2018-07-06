#!/usr/bin/env python

import os

import numpy as np
import pylab as plt

import dvpy as dv

mdir = os.path.expanduser('~/Dropbox/Cardiac_Segmentation/manuscript-ohm-net-data/')
tdir = os.path.join(mdir, 'transform_mat')
opacity = 0.02
o_x = 0.05
o_y = 0.85

view = [np.load(os.path.expanduser('~/Developer/Bitbucket/semanticsegmentation/data/img_list_%d.npy'%(i))) for i in range(3)]
view = np.concatenate(view)

c_dict = {'00_SAX' : 'orange',
          '01_HLA' : 'green',
          '02_VLA' : 'violet',
         }
view = [dv.tokenize_path(v)[-4] for v in view]
view = [c_dict[v] for v in view]

## Translation

xy_t = [np.load(os.path.join(tdir,'true_xy_mat_batch_%d_hgd_3.npy'%(i))) for i in range(3)]
xy_p = [np.load(os.path.join(tdir,'prediction_xy_mat_batch_%d_hgd_3.npy'%(i))) for i in range(3)]
xy_t = np.concatenate(xy_t)
xy_p = np.concatenate(xy_p)
xy_a = (xy_t + xy_p) / 2.0
xy_d = xy_t - xy_p

## Rotation

rt_t = [np.load(os.path.join(tdir,'true_rt_mat_batch_%d_hgd_3.npy'%(i))) for i in range(3)]
rt_p = [np.load(os.path.join(tdir,'prediction_rt_mat_batch_%d_hgd_3.npy'%(i))) for i in range(3)]
rt_t = np.concatenate(rt_t) + np.pi
rt_p = np.concatenate(rt_p) + np.pi
rt_t = dv.wrap_phase(rt_t).squeeze()
rt_p = dv.wrap_phase(rt_p).squeeze()
rt_a = dv.wrap_phase(rt_t + rt_p) / 2.0
rt_d = dv.wrap_phase(rt_t - rt_p)

## Zoom

zm_t = [np.load(os.path.join(tdir,'true_zm_mat_batch_%d_hgd_3.npy'%(i))) for i in range(3)]
zm_p = [np.load(os.path.join(tdir,'prediction_zm_mat_batch_%d_hgd_3.npy'%(i))) for i in range(3)]
zm_t = np.concatenate(zm_t).squeeze()
zm_p = np.concatenate(zm_p).squeeze()
zm_a = (zm_t + zm_p) / 2.0
zm_d = zm_t - zm_p

## Build Figure

fg = plt.figure(figsize=(14, 7))

ax_c_x = fg.add_subplot(2, 4, 1, aspect = 'equal')
ax_c_y = fg.add_subplot(2, 4, 2, aspect = 'equal')
ax_c_r = fg.add_subplot(2, 4, 3, aspect = 'equal')
ax_c_z = fg.add_subplot(2, 4, 4, aspect = 'equal')

ax_b_x = fg.add_subplot(2, 4, 5)
ax_b_y = fg.add_subplot(2, 4, 6)
ax_b_r = fg.add_subplot(2, 4, 7)
ax_b_z = fg.add_subplot(2, 4, 8)

ax_b_x.set_aspect('equal', 'datalim')
ax_b_y.set_aspect('equal', 'datalim')
ax_b_r.set_aspect('equal', 'datalim')
ax_b_z.set_aspect('equal', 'datalim')

## Correlations

ax_c_x.set_title('Translation (Horizontal)')
mn = -0.45
mx = +0.45
ax_c_x.scatter(xy_t[:,0], xy_p[:,0], alpha = opacity, c = view)
ax_c_x.set_xlim([mn, mx])
ax_c_x.set_ylim([mn, mx])
ax_c_x.plot([mn, mx], [mn, mx], '--', color = 'gray')
dv.add_trendline(ax_c_x, xy_t[:,0], xy_p[:,0])
annotation = dv.annotate_linear_regression(xy_t[:,0], xy_p[:,0], x_label = r'\hat{t}_x', y_label = r't_x')
ax_c_x.text(o_x, o_y, annotation, transform = ax_c_x.transAxes)
ax_c_x.set_xlabel(r'$\hat{t}_x$')
ax_c_x.set_ylabel(r'$t_x$')

ax_c_y.set_title('Translation (Vertical)')
mn = -0.45
mx = +0.45
ax_c_y.scatter(xy_t[:,1], xy_p[:,1], alpha = opacity, c = view)
ax_c_y.set_xlim([mn, mx])
ax_c_y.set_ylim([mn, mx])
ax_c_y.plot([mn, mx], [mn, mx], '--', color = 'gray')
dv.add_trendline(ax_c_y, xy_t[:,1], xy_p[:,1])
annotation = dv.annotate_linear_regression(xy_t[:,1], xy_p[:,1], x_label = r'\hat{t}_y', y_label = r't_y')
ax_c_y.text(o_x, o_y, annotation, transform = ax_c_y.transAxes)
ax_c_y.set_xlabel(r'$\hat{t}_y$')
ax_c_y.set_ylabel(r'$t_y$')

ax_c_r.set_title('Rotation')
mn = -np.pi - 0.1
mx = 1.8
ax_c_r.scatter(rt_t, rt_p, alpha = opacity, c = view)
ax_c_r.set_xlim([mn, mx])
ax_c_r.set_ylim([mn, mx])
ax_c_r.plot([mn, mx], [mn, mx], '--', color = 'gray')
dv.add_trendline(ax_c_r, rt_t, rt_p)
annotation = dv.annotate_linear_regression(rt_t, rt_p, x_label = r'\hat{\theta}', y_label = r'\theta')
ax_c_r.text(o_x, o_y, annotation, transform = ax_c_r.transAxes)
ax_c_r.set_xlabel(r'$\hat{\theta}$')
ax_c_r.set_ylabel(r'$\theta$')

ax_c_z.set_title('Scale')
mn = 0.3
mx = 1.1
ax_c_z.scatter(zm_t, zm_p, alpha = opacity, c = view)
ax_c_z.set_xlim([0.3, 1.1])
ax_c_z.set_ylim([0.3, 1.1])
ax_c_z.plot([mn, mx], [mn, mx], '--', color = 'gray')
dv.add_trendline(ax_c_z, zm_t, zm_p)
annotation = dv.annotate_linear_regression(zm_t, zm_p, x_label = r'\hat{s}', y_label = r's')
ax_c_z.text(o_x, o_y, annotation, transform = ax_c_z.transAxes)
ax_c_z.set_xlabel(r'$\hat{s}$')
ax_c_z.set_ylabel(r'$s$')

## Bland-Altmans
o_y = 0.8

ax_b_x.scatter(xy_a[:,0], xy_d[:,0], alpha = opacity, c = view)
print(1.96*np.std(xy_d[:,0]))
dv.add_horizontal_limits(ax_b_x, np.mean(xy_d[:,0]), np.std(xy_d[:,0]))
annotation = dv.annotate_bland_altman(xy_t[:,0], xy_p[:,0], x_label = r'$\hat{t}_x$', y_label = r'$t_x$')
ax_b_x.text(o_x, o_y, annotation, transform = ax_b_x.transAxes)
ax_b_x.set_xlabel(r'$(\hat{t}_x + t_x) / 2$')
ax_b_x.set_ylabel(r'$\hat{t}_x - t_x$')

ax_b_y.scatter(xy_a[:,1], xy_d[:,1], alpha = opacity, c = view)
print(1.96*np.std(xy_d[:,1]))
dv.add_horizontal_limits(ax_b_y, np.mean(xy_d[:,1]), np.std(xy_d[:,1]))
annotation = dv.annotate_bland_altman(xy_t[:,1], xy_p[:,1])
ax_b_y.text(o_x, o_y, annotation, transform = ax_b_y.transAxes)
ax_b_y.set_xlabel(r'$(\hat{t}_y + t_y) / 2$')
ax_b_y.set_ylabel(r'$\hat{t}_y - t_y$')

ax_b_r.scatter(rt_a, rt_d, alpha = opacity, c = view)
print(1.96*np.std(rt_d))
dv.add_horizontal_limits(ax_b_r, np.mean(rt_d), np.std(rt_d))
annotation = dv.annotate_bland_altman(rt_t, rt_p)
ax_b_r.text(o_x, o_y, annotation, transform = ax_b_r.transAxes)
ax_b_r.set_xlabel(r'$\mathcal{W}(\hat{\theta} + \theta) / 2$')
ax_b_r.set_ylabel(r'$\mathcal{W}(\hat{\theta} - \theta)$')

ax_b_z.scatter(zm_a, zm_d, alpha = opacity, c = view)
print(1.96*np.std(zm_d))
dv.add_horizontal_limits(ax_b_z, np.mean(zm_d), np.std(zm_d))
annotation = dv.annotate_bland_altman(zm_t, zm_p)
ax_b_z.text(o_x, o_y, annotation, transform = ax_b_z.transAxes)
ax_b_z.set_xlabel(r'$(\hat{s} + s) / 2$')
ax_b_z.set_ylabel(r'$\hat{s} - s$')

plt.tight_layout()
plt.savefig(os.path.expanduser('~/Dropbox/Cardiac_Segmentation/manuscript-ohm-net-data/matrix-loss.png'), dpi = 300)

