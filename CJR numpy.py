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



#Let's make some arrays

import numpy as np

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

a-b



















##Now let's simulate some brain imaging data
random_brain = np.random.normal(0,1, size =(100,100,100))
print("The dimensions of voxels in this simulated brain are:", random_brain.shape)

#Now let's index a random voxel of our simulated brain
print("The value of the voxel at dimensions 10,24,50 is equal to :", random_brain[10,24,50])

#Let's take a slice out of the three dimensional image
print(random_brain[:,:,3])







