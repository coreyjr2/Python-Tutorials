#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 10:54:22 2020

@author: cjrichier
"""

'''This is a series of tutorials using the NiLearn library'''

#Load libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting, datasets
import pandas as pd

##let's do some basic functions first. 
plotting.plot_glass_brain(dataset)









#Load the dataset, just one subject so it doesn't crash your computer
dataset = datasets.fetch_development_fmri(n_subjects=1)
func_filename = dataset.func[0]
confound_filename = dataset.confounds[0]


#Let's set some coordinates of the posterior cingulate cortex as our seed sphere
pcc_coords = [(0, -52, 18)]


'''Now let's extract the time series from this particular area of the brain.
The center of the sphere are the coordinates we selected at the PCC, and we will 
set the radius to be 8 voxels around this point. We will also apply filters 
to the series.
'''

from nilearn import input_data

seed_masker = input_data.NiftiSpheresMasker(
    pcc_coords, radius=8,
    detrend=True, standardize=True,
    low_pass=0.1, high_pass=0.01, t_r=2,
    memory='nilearn_cache', memory_level=1, verbose=0)

'''Now let's regress out confounds for this voxel found in the dataset'''

seed_time_series = seed_masker.fit_transform(func_filename, confounds=[confound_filename])

'''Now let's plot the time series of this region of the brain'''

#In order to plot, we have to make it a dataframe object with pandas first
seed_ts_df = pd.DataFrame(seed_time_series)
seed_ts_df.plot()
plt.title('Seed time series (Posterior cingulate cortex)')
plt.xlabel('Scan number')
plt.ylabel('Normalized signal')
plt.tight_layout()
plt.show()
'''Nice. Looks like something is happening in the brain at this region. 
Who would have thought.

Let's try it out for the whole brain, this time by each voxel.'''
brain_masker = input_data.NiftiMasker(
    smoothing_fwhm=6,
    detrend=True, standardize=True,
    low_pass=0.1, high_pass=0.01, t_r=2,
    memory='nilearn_cache', memory_level=1, verbose=0)
brain_time_series = brain_masker.fit_transform(func_filename, confounds=[confound_filename])

'''brain_time_series is an object of dimensions 168x32504.
this means we have 168 time points (as made evident by out 
last graph of the single region) and 32504 voxels with a time series for 
each. Establishing and understanding the structure of your neuroimaging
data is really important, so try to always orient yourself to the shape 
of what you are working with.'''

''' let's take a look at some random voxels. For example, we will use 5.'''


brain_ts_df = pd.DataFrame(brain_time_series)
plt.plot(brain_ts_df.iloc[:, [3, 57, 846, 2967, 12345]])
plt.xlabel('Scan number')
plt.ylabel('Normalized signal')
plt.tight_layout()

'''Now let's compare this seed region to the timer series in every voxel'''
seed_to_voxel_correlations = (np.dot(brain_time_series.T, seed_time_series) /seed_time_series.shape[0])
'''this makes an array of the correlation between every voxel to the seed region'''

'''Now let's visualize how this might appear in the brain'''
seed_to_voxel_correlations_img = brain_masker.inverse_transform(seed_to_voxel_correlations.T)
display = plotting.plot_stat_map(seed_to_voxel_correlations_img,threshold=0.5, vmax=1,cut_coords=pcc_coords[0], title="Seed-to-voxel correlation (PCC seed)")
display.add_markers(marker_coords=pcc_coords, marker_color='g',marker_size=300)
# At last, we save the plot as pdf.
display.savefig('pcc_seed_correlation.pdf')



