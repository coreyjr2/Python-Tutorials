#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 12:31:59 2020

@author: cjrichier
"""

'''This is a tutorial on Pandas. it works a lot like the 
functionality in base R for dataframes'''

#import the library
import pandas as pd

#find working directory
import os
print(os.getcwd())

#Change working directory (in the event it is not where we want it)
os.chdir("/Users/cjrichier/Documents/GitHub/")

#now let's import some data
#using relative path (that is, easier to generalize across machines)
data = pd.read_csv('Python-Tutorials/afq_kmc.csv', sep='\t')

data.head()
data.values


#how to index in pandas, while also summarizing data
data.iloc[:, :3].describe

#index a column by position (return all of the first column for me)
data.iloc[:, 0]

