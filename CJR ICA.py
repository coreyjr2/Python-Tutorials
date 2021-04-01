#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 17:44:00 2020

@author: cjrichier
"""

'''Load in some relevant libraries'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
import json

import nibabel as nib

def load_image(path):
    # load an img file
    return nib.load(path)

def get_TR(img):
    # retrieve TR data
    return img.header.get_zooms()[-1]

def get_slices(img):
    # retrieve number of slices
    return img.shape[2]
  
def get_header(img):
    # print the full header
    return(img.header)

'''Let's do a feature extraction now'''

from nilearn import input_data
from nilearn import datasets
from nilearn import plotting
from nilearn.plotting import plot_prob_atlas, plot_roi, plot_matrix

from nilearn.decomposition import CanICA 
from nilearn import image
from nilearn.regions import RegionExtractor


from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure

correlation_measure = ConnectivityMeasure(kind='correlation')

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")


'''upload 4 fMRI scans files, one for each diagnosis.'''
rest_img_adhd_sub_70001=load_image('/Volumes/Byrgenwerth/Datasets/UCLA Consortium for Neuropsychiatric Phenomics LA5c Study/sub-70001/func/sub-70001_task-rest_bold.nii.gz')
rest_img_bp_sub_60001=load_image('/Volumes/Byrgenwerth/Datasets/UCLA Consortium for Neuropsychiatric Phenomics LA5c Study/sub-60001/func/sub-60001_task-rest_bold.nii.gz')
rest_img_sz_sub_50006=load_image('/Volumes/Byrgenwerth/Datasets/UCLA Consortium for Neuropsychiatric Phenomics LA5c Study/sub-50006/func/sub-50006_task-rest_bold.nii.gz')
rest_img_cn_sub_10228=load_image('/Volumes/Byrgenwerth/Datasets/UCLA Consortium for Neuropsychiatric Phenomics LA5c Study/sub-10228/func/sub-10228_task-rest_bold.nii.gz')


'''these are the test subjects'''
rest_img_adhd_sub_70079=load_image('/Volumes/Byrgenwerth/Datasets/UCLA Consortium for Neuropsychiatric Phenomics LA5c Study/sub-70079/func/sub-70079_task-rest_bold.nii.gz')
rest_img_bp_sub_60073=load_image('/Volumes/Byrgenwerth/Datasets/UCLA Consortium for Neuropsychiatric Phenomics LA5c Study/sub-60073/func/sub-60073_task-rest_bold.nii.gz')
rest_img_sz_sub_50067=load_image('/Volumes/Byrgenwerth/Datasets/UCLA Consortium for Neuropsychiatric Phenomics LA5c Study/sub-50067/func/sub-50067_task-rest_bold.nii.gz')
rest_img_cn_sub_10249=load_image('/Volumes/Byrgenwerth/Datasets/UCLA Consortium for Neuropsychiatric Phenomics LA5c Study/sub-10249/func/sub-10249_task-rest_bold.nii.gz')




'''let's check on one subject just to see how it works'''
print(rest_img_cn_sub_10228)
rest_img_cn_sub_10228.shape
'''What are the dimensions of these images?'''


'''These are the raw fMRI files, we will need them to get two things. 
First, an understanding of what regions are what.
Secondly, we can extract the time series of these data as well. 
'''

'''First we will build a masker. Maskers are used 
frequently in neuroimaging data analysis to process raw image filea
to get them to do what we want them to do. More specifically, 
the shape of data in its current state, a 4D series of images,
isn't really im the best shape for building statistical models,
so we will need to do some finagling to get it in the shape we 
need it to be in. Maskers can also filter fMRI data to extract 
only the parts we care about.'''


## craete masker based on the atlas 
## and create a time series of the uploaded image using the masker
def create_mask(atlas_img, fmri_img):
  # generates a mask given img and atlas
  masker=NiftiLabelsMasker(labels_img=atlas_img, standardize=True)
  time_series=masker.fit_transform(fmri_img)
  
  return time_series

# using the correlation measures defined above, 
# we calculate the correaltion matrixes
def calc_correlation_matrix(time_series):
  # given a time series, return a correlation matrix
  return correlation_measure.fit_transform([time_series])[0]

#and we plot,
def plot_cor_matrix(correlation_matrix, title, labels=None):
  ## plot the correlation matrix
  
    
  np.fill_diagonal(correlation_matrix, 0)
  if labels:
    plot_matrix(correlation_matrix, figure=(10, 8), 
              labels=labels,
                       vmax=0.8, vmin=-0.8, reorder=True)
  else:
    plot_matrix(correlation_matrix, figure=(10, 8), 
              labels=range(correlation_matrix.shape[1]),
                       vmax=0.8, vmin=-0.8, reorder=True)
  plt.title(title)
  plt.show()



'''Now we will load in an atlas to define what region goes where. This can be accomplished using many different atlases.
For the purposes of this tutorial, we will use the MSDL Atlas.'''
## import an existing map
msdl = datasets.fetch_atlas_msdl()
'''Now let's extract some important attributes from this parcellation, notably maps, labels, and netowrk information'''
msdl_maps = msdl.maps
msdl_labels = msdl.labels
msdl_networks = msdl.networks
msdl_coordinates = msdl.region_coords

'''Now let's build our masker'''
from nilearn.input_data import NiftiMapsMasker


'''create masker to extract functional data within atlas parcels'''
masker = NiftiMapsMasker(maps_img=msdl['maps'], standardize=True,
                         memory='nilearn_cache')

'''calculate the correlation matrix for each of the four subjects'''
#Control
rest_cn = np.array(masker.fit_transform(rest_img_cn_sub_10228))
cn_matrix = calc_correlation_matrix(rest_cn)
#Schizophrenia
rest_sz = np.array(masker.fit_transform(rest_img_sz_sub_50006))
sz_matrix = calc_correlation_matrix(rest_sz)
#ADHD
rest_adhd = np.array(masker.fit_transform(rest_img_adhd_sub_70001))
adhd_matrix = calc_correlation_matrix(rest_adhd)
#Bipolar
rest_bp = np.array(masker.fit_transform(rest_img_bp_sub_60001))
bp_matrix = calc_correlation_matrix(rest_bp)


'''calculate the correlation matrix for the test subjects'''
test_rest_cn = np.array(masker.fit_transform(rest_img_cn_sub_10249))
test_cn_matrix = calc_correlation_matrix(rest_cn)
#Schizophrenia
test_rest_sz = np.array(masker.fit_transform(rest_img_sz_sub_50067))
test_sz_matrix = calc_correlation_matrix(rest_sz)
#ADHD
test_rest_adhd = np.array(masker.fit_transform(rest_img_adhd_sub_70079))
test_adhd_matrix = calc_correlation_matrix(rest_adhd)
#Bipolar
test_rest_bp = np.array(masker.fit_transform(rest_img_bp_sub_60073))
test_bp_matrix = calc_correlation_matrix(rest_bp)



from nilearn.connectome import ConnectivityMeasure

correlation_measure = ConnectivityMeasure(kind='correlation')

from nilearn import plotting
from matplotlib import pyplot as plt
plotting.plot_matrix(test_bp_matrix , tri='lower', colorbar=False, title='correlation')


'''let's plot the coorelation matrices for each subject'''
plot_cor_matrix(cn_matrix, title="Healthy Control")
plot_cor_matrix(sz_matrix, title="Schizophrenia")
plot_cor_matrix(adhd_matrix, title="ADHD")
plot_cor_matrix(bp_matrix, title="Bipolar")


'''Now let's plot and compare all of the different subjects'''
plotting.plot_connectome(cn_matrix, msdl_coordinates,
                         edge_threshold="80%", title='Healthy Control')
plotting.plot_connectome(sz_matrix, msdl_coordinates,
                         edge_threshold="80%", title='Schizophrenia')
plotting.plot_connectome(adhd_matrix, msdl_coordinates,
                         edge_threshold="80%", title='ADHD')
plotting.plot_connectome(bp_matrix, msdl_coordinates,
                         edge_threshold="80%", title='Bipolar')
plotting.show()


'''Now we are going to pull in some clinical and behavioral data.'''
scid = pd.read_table(r'/Volumes/Byrgenwerth/Datasets/UCLA Consortium for Neuropsychiatric Phenomics LA5c Study/phenotype/scid.tsv')
demo = pd.read_table(r'/Volumes/Byrgenwerth/Datasets/UCLA Consortium for Neuropsychiatric Phenomics LA5c Study/phenotype/demographics.tsv')

'''Take only the participants we need'''
scid_short = scid.loc[(scid['participant_id']== 'sub-10228') | (scid['participant_id']== 'sub-50006')
         | (scid['participant_id']== 'sub-60001') | (scid['participant_id']== 'sub-70001')]



'''let's flatten the correlation matrices'''
#cn_vector = np.tril(cn_matrix)
#cn_vector = cn_vector.flatten()
#cn_vector = cn_vector[cn_vector != 0]
#cn_vector = cn_vector[cn_vector != 1]

from nilearn.image.image import mean_img
mean_cn= mean_img(rest_img_cn_sub_10228)
from nilearn.plotting import plot_epi, show
plot_epi(mean_cn)
         
         
'''Assemble all FC matrices into one object'''
from nilearn.connectome import sym_matrix_to_vec
all_fc = np.stack((sz_matrix, bp_matrix, cn_matrix, adhd_matrix, test_sz_matrix, test_bp_matrix, test_adhd_matrix, test_cn_matrix))

#concat all the TS into one object
concat_ts = np.stack((rest_cn, rest_sz,rest_bp,rest_adhd, 
                test_rest_cn, test_rest_sz, test_rest_bp, test_rest_adhd))


concat_ts = np.asarray(concat_ts)

connectivity = ConnectivityMeasure(kind='correlation', vectorize=True)
connectivity.fit_transform(concat_ts)

for subject in all_fc:
   sym_matrix_to_vec(subject, discard_diagonal=True)
        
   
    
sz_vector = sym_matrix_to_vec(sz_matrix, discard_diagonal=True)
bp_vector = sym_matrix_to_vec(bp_matrix, discard_diagonal=True)
adhd_vector = sym_matrix_to_vec(adhd_matrix, discard_diagonal=True)
cn_vector = sym_matrix_to_vec(cn_matrix, discard_diagonal=True)
test_sz_vector = sym_matrix_to_vec(test_sz_matrix, discard_diagonal=True)
test_bp_vector = sym_matrix_to_vec(test_bp_matrix, discard_diagonal=True)
test_adhd_vector = sym_matrix_to_vec(test_adhd_matrix, discard_diagonal=True)
test_cn_vector = sym_matrix_to_vec(test_cn_matrix, discard_diagonal=True)
   
concat_vector = np.stack((sz_vector, bp_vector,adhd_vector ,cn_vector, 
                test_sz_vector, test_bp_vector, test_adhd_vector, test_cn_vector))


data = pd.DataFrame(concat_vector)


diagnosis = np.array([1,2,3,4,1,2,3,4])

data['diagnosis'] = diagnosis

connectivity = ConnectivityMeasure(kind='correlation', vectorize=True)
for subject in concat_ts:
    connectivity.fit_transform(subject)
    

'''Now let's try to build the model'''
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score













# perform an ICA given the subset of the data. 
ica = CanICA(n_components=20,
                random_state=0)

ica.fit(img)
components_img=ica.components_img_
plot_prob_atlas(components_img, title='All ICA components')
plt.show()

components_img_1st=image.index_img(components_img, 0)
ica_time_series=create_mask(components_img_1st, img)
ica_cor_matrix=calc_correlation_matrix(ica_time_series)
plot_cor_matrix(ica_cor_matrix, 'ICA correlation matrix')




'''Now let's take a look at the Schizophrenic subject to compare'''
path_sz = '/Volumes/Byrgenwerth/Datasets/UCLA Consortium for Neuropsychiatric Phenomics LA5c Study/sub-50006/func/sub-50006_task-rest_bold.nii.gz'
img_sz= load_image(path_sz)

# ica SZ (same componenets as control)

components_img_sz_1st=image.index_img(components_img, 0)
ica_sz_time_series=create_mask(components_img_sz_1st, img_sz)
ica_sz_cor_matrix=calc_correlation_matrix(ica_sz_time_series)
plot_cor_matrix(ica_sz_cor_matrix, 'ICA correlation matrix- SZ')
'''interesting, they look quite different!'''


'''let's correlate both subjects with each other now'''


ica_sz_time_series=create_mask(components_img_sz_1st, img_sz)
ica_sz_cor_matrix=calc_correlation_matrix(ica_sz_time_series)
diff_cor_mtarix = ica_cor_matrix-ica_sz_cor_matrix


























