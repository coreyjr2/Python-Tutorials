#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 14:59:41 2020

@author: cjrichier
"""


'''Load some relevant libraries'''
import os
import nibabel as nib
import nilearn as nl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''import the data'''
haxby_data =  nl.datasets.fetch_haxby(data_dir="/Volumes/Byrgenwerth/Datasets/Haxby/")
haxby_data_DIR = "/Volumes/Byrgenwerth/Datasets/Haxby/haxby2001"
if not os.path.isdir(HCP_DIR): os.mkdir(HCP_DIR)


'''now let's take a look at what this data has 
in the files we just downloaded'''
list(sorted(haxby_data.keys())) 
print(haxby_data['description']) 
 

'''Now let's load in some behavioral data'''
beh_data = pd.read_csv(haxby_data.session_target[0], delimiter=' ')
print(beh_data)

'''Let's only work with a few certain conditions to start.
Right now, the cat and face conditons'''
list(sorted(beh_data.keys())) 

fmri_filename = haxby_data.func[0]
# print basic information on the dataset
print('First subject functional nifti images (4D) are at: %s' %
      fmri_filename)  # 4D data

'''Let's visualize some fMRI volumes. One way to do this is 
to use nilearn. fmri data is 4D so we have to compress
it down to 3D dimensions in order to visualize using 
this function because it only accepts 3D data'''

from nilearn import plotting
from nilearn.image import mean_img
plotting.view_img(mean_img(fmri_filename), threshold=None)

'''Now we need to convert the data into matricies 
in order to apply ML to the data. We do this by 
applying a masker object. Maskers basically take
the 4D data and compress rearrange them into 2D. 
For example, if we have a group of voxels we think
are in the Amygdala, the masker will extract them from 
the time series of 3D images (4D series) and arrange them
to be a data matrix with the voxel values as rows and
the columns as time points, for example.''' 
mask_filename = haxby_data.mask_vt[0]
plotting.plot_roi(mask_filename, bg_img=haxby_data.anat[0],
                 cmap='Paired')
'''What we have done here is plotted the voxels that we 
have applied the mask to agains the backdrop of the
actual brain volume'''
from nilearn.input_data import NiftiMasker
masker = NiftiMasker(mask_img=mask_filename, standardize=True)
fmri_masked = masker.fit_transform(fmri_filename)
masker.generate_report()
print(fmri_masked)
print(fmri_masked.shape)
'''as you can see, the object fmri_masked is 1452x464,
which is the number of time points by the number of voxels.'''

'''Now why don't we do a little bit of visualization. 
What does the time series for some random voxels look like?'''
plt.plot(fmri_masked[200:400, 45:52])
plt.title('Time Series for Some Random Masked Voxels')
plt.xlabel('Time')
plt.ylabel('BOLD signal')
plt.tight_layout()
'''Pretty cool, huh? Now we're doing science!'''


'''Now let's circle back around to that behavioral
data. Now that we have arranged the brain data in 
a 2D, this will make it play nicer with the behavioral 
data we have. Let's take a peek at it again'''
print(beh_data)
'''If you notice, this data has the same number of rows
as the imaging data -- corresponding to time points.
Nice! That makes this much easier to conceptualize.'''

'''There were various conditions done in the task (as is the case
in many experiments.) Let's try and predict what task they were
doing based on the pattern of activity in the region
we masked out'''
conditions = beh_data['labels']
conditions
'''So now we have an object that we can use as our dependent
variable. So now, the regression model might be taking shape
if you are able to picture it. '''

'''Now the subjects saw a variety of different conditions.
Let's take a peek at what they all were'''
conditions.unique()
'''Some interesting categories here. Perhaps we can choose shoes and scissors.
They are seemingly distinct.'''

'''Selecting some certain coditions lets us shrink the data
as well. Let's give that a shot by applying a mask for the conditions and
to our dependent variable'''

condition_mask = conditions.isin(['scissors', 'shoe'])
fmri_masked = fmri_masked[condition_mask]
print(fmri_masked.shape)
conditions = conditions[condition_mask]
print(conditions.shape)
'''There we go, now the data is a lot smaller. it is only 216 time points now
for both the input and output of our model.'''

'''Now it's time for the machine learning. Are you ready for your machine to 
learn ya some data? We will use Support Vector Classification in this case.'''
from sklearn.svm import SVC
svc = SVC(kernel='linear')
print(svc)

'''Now that we have fit the model, let's run it on the data'''
svc.fit(fmri_masked, conditions)
'''and test the model's predictive ability'''
prediction = svc.predict(fmri_masked)
print(prediction)
print((prediction == conditions).sum() / float(len(conditions)))

'''Now we didn't do any cross validation, so let's do that this time'''
from sklearn.model_selection import cross_val_score
session_label = beh_data['chunks'][condition_mask]
cv_score = cross_val_score(svc, fmri_masked, conditions, cv=cv)
print(cv_score)
mean_model_accuracy = np.mean(cv_score)
print(mean_model_accuracy)
'''So after five folds, it ooks like our accuracy is about 76%. Not too bad!'''

'''Let's take a peek at the weights that were applied to every voxel in the model'''
coef_ = svc.coef_
print(coef_)
'''that's understandably a lot of numbers are hard to make much sense of.
Let's graph it isntead. This can be done by converting it into a Nifti image.'''
coef_img = masker.inverse_transform(coef_)
print(coef_img)
coef_img.to_filename('haxby_svc_weights.nii.gz')
from nilearn.plotting import plot_stat_map, show
plot_stat_map(coef_img, bg_img=haxby_data.anat[0],
              title="SVM weights", display_mode="yx")
show()
'''Neat, so this is a graphical representation of the voxels that have weights
applied to them. Higher weights/lower are applied to more colored voxels.'''






