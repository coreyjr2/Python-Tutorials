#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 12:45:27 2020

@author: cjrichier
"""

'''This is a NumPy tutorial, meant for 
teaching some of the basics to anyone who 
wants to learn more about the applications of python for scientific purposes.
I had somewhat of a hard time getting started, so in whatever way I can make the process 
easier for those who are curious, I am happy to do so. Open science is just 
as much about open education as it is about all of the other aspects of the job!'''


'''Python is a very general programming language, so while that makes it very flexible,
it does not have some of the same base functionality for scientific data analysis that
languages like R or MATLAB may have. This is not to say it cannot accomplish 
the same functionality, it just is not as efficient. As such, many libraries, 
such as NumPy, are great tools for utilizing Python to help analyze scientific data. 
This tutorial will explain some of the basics of arrays and mathematical operations with 
arrays'''


#Let's make some arrays

import numpy as np
import matplotlib as plt

my_list = ([1,2,3],[4,5,6],[7,8,9])
my_array = np.array(my_list)
print(my_array)


#Let's make a zero array 
array_3d = np.zeros([5,5,5])
print(array_3d)

random_array = np.random.normal(0,1, size =(5,5))
print(random_array)

#Let's look at the shape of objects
print("The shape of this array is:", array_3d.shape)



#Let's do some matrix operations on the arrays
a = np.array([1,2,3,4,5])
b = np.array([3,4,5,8,1])


#subtract the values of a from b
a-b

#Let's take the natural log of each element of array a
np.log(a)

###Let's do some basic linear algebra###
x = np.random.normal(0,1, size =(4,4,4))
np.linalg.inv(x)

#Let's make an identity matrix (of dimensions 5x5)
np.eye(5)

#Let's find the inverse of a matrix
np.linalg.inv()


## some other stuff related to array broadcasting
arr_2d = np.zeros([5,5])
arr_ones = np.ones([3,3])
arr_2d[:3,2:] = arr_ones
arr_2d

'''What we did here was nest the array of ones (3x3) inside a 
another array of zeros with dimensions 5x5'''






#Let's do some more indexing with arrays

#make a 4x4 array
x = np.reshape(np.arange(16), (4,4))
print("Let's index this array:", x)

i = np.array([0,3])
j = np.array([1,1])



##Now let's simulate some brain imaging data

'''#Now, this isnt *really* like what you would have in an image, because it is lacking
#information from the time dimension, and we have values in every voxel, as 
#opposed to only where there would be actual brain tissue, but it will suffice 
#for now


random_brain = np.random.normal(0,1, size =(100,100,100))
print("The dimensions of voxels in this simulated brain are:", random_brain.shape)

#Now let's index a random voxel of our simulated brain
print("The value of the voxel at dimensions 10,24,50 is equal to :", random_brain[10,24,50])

#Let's take a slice out of the three dimensional image
print(random_brain[:,:,3])


### Let's do some data filtering on the basis of boolean values
is_big = random_brain > 2
big_values = random_brain[is_big]
big_values.shape
#So it looks like 22676 values are over 2 standard deviations above the mean

#Copy the original image to prevent modification
filtered_brain = random_brain.copy()

#set all voxel values greater than 2sd to 0
filtered_brain[~is_big] = 0
filtered_brain.shape
# the tilde ~ means "not" in this case

##let's do some plotting
import matplotlib.pyplot as plt

#take a slice of each
orig_slice = random_brain[:, :, 45]
filter_slice = filtered_brain[:, :, 45]


#Now let's compare original and thresholded image
fig, axes = plt.subplots(1, 2, figsize=(4,10))
axes[0].imshow(orig_slice)
axes[1].imshow(filter_slice)

#Some tips on vectorization
#looping over some values
x = np.arange(5)
y = np.zeros((5,))
for i in range (5):
    y[i] = x[i] + 1
y

#or you could do this
x + 1
#The latter way is much faster
#Try to vectorize your code always!!






