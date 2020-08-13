#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 13:48:43 2020

@author: cjrichier
"""

'''This is a tutorial for NiBabel, which is used to maniuplate and open various
formats of neuroimaging data.'''


#import the package
import nibabel as nb

#find working directory
import os
print(os.getcwd())

#Change working directory (in the event it is not where we want it)
os.chdir("/Users/cjrichier/Documents/GitHub/Python-Tutorials/")

#Let's load in some example data
anat = nb.load('tutorial_anatomy.nii.gz')
epi = nb.load('tutorial_epi.nii.gz')

epi_img = epi.get_fdata()
anat_img = anat.get_fdata()

#What are the dimensions of our data?
anat.shape
epi.shape

#now let's take a peek at what we are dealing with
import matplotlib.pyplot as plt

'''define a function to display some image slices:'''
def show_slices(slices):
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
        
#now let's take an arbitary axial, coronal, and saggital slice 
saggital_boi = epi_img[30, :, :]
coronal_boi = epi_img[:, 31, :]
axial_boi = epi_img[:, :, 32]
show_slices([saggital_boi, coronal_boi, axial_boi])

#Now for the anat images
show_slices([anat_img[28, :, :], anat_img[:, 33, :], anat_img[:, :, 28]])


'''Voxel coordinates are coordinates in the image array 
(in this case a 3d image for brains) in the case above, we took 3 2d dimensional
slices (of the saggital, coronal, and axial perspectives) where the data "values"
represent the greyscale intensitiy of a given voxel. Hopefully this is clear when you look 
at the picture. Some pixels are dark (or even black) and others are lighter (close to white)
This, in essence, is how we can reconstruct the image in terms of an array of numerical values
into what really looks like a brain. Super neat, right?'''

#next, let's try an extract the value for a single voxel. So not a picture, jsut a point.
n_i, n_j, n_k = epi_img.shape
center_i = (n_i - 1) // 2  # // for integer division
center_j = (n_j - 1) // 2
center_k = (n_k - 1) // 2
center_i, center_j, center_k
(26, 30, 16)
center_vox_value = epi_img[center_i, center_j, center_k]
center_vox_value
#So, the greyscale value of the voxel at position (26, 30, 16) is 81.54928779602051

'''So that's nice that we can extract data from the image. But there's more to it.
If we were to superimpose the epi and anatomical image, we would see that they 
would be overlapping in 3d space... An example. If we want to reference 
the epi and anatomical images to one another, a voxel that is the amygdala in one 
might end up being in the motor cortex in another. Tt doesn't take a lot
of knowledge about brain anatomy to see how that would be big bad'''

'''Luckily, with some pretty clever applications of linear algebra, we can shift the images
in space to be aligned with one another. First, we will need to create what's called a reference
space. The reference space is used to map both images to some common ground'''


###code for spatial transformations here

