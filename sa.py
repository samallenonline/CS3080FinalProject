# CS3080 - Final Project 
# Gender transitioning and changes in self-reported sexual orientation

# using this tutorial to help me out --> https://www.w3schools.com/python/python_ml_multiple_regression.asp

# import libraries 
import pandas as pd 
import numpy as np 
#import matplotlib.pyplot as plt # Optional to use
#import seaborn as sns # Optional to use

# from scikit-learn 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

print("* All libraries have been successfull imported")

# load in data
# LOE: data_simplified_preclean.csv is data_simplified but
#      without all NA/empty values manually replaced to be 0.
#      It is the same as sexual_orientation_auer_anonymized.csv,
#      but without the rows of empty cells
data = pd.read_csv("data_simplified_preclean.csv")

# list of independent variables 
'''
# LOE: Copied and simplified below for basic testing purposes
# LOE: Saved for later if needed
X = data[['sex (1=MtF; 2 =FtM)',
'initial_sex_orientation (1= androphilic; 2 =gynephilic; 3 = bisexual, 4 = analloerotic)', 
'age_of_onset',
'onset_before_age_of_12 (1= before or at age of 12; after age of 12)',
'age_psychol (age of first psychological counselling)',
'age_role (age of start everday-experience)',
'hormontherapy (1 =yes; 2 =no)',
'age_hormonetherapy (age of initiation of hormonetherapy)',
'hormonetype (1= T transdermal; 2 = T intramuscular; 3 = E + Antiandrogen; 4 = Estradiol transdermal; 5 = Estradiol oral; 6 = estradiol +gestagen)',
'sex reassignement surgery (1= yes; 2 = no)',
'age_surgery (age of sex reassignement surgery)', 
'type_of_surgery (1 = hysterectomy + mastectomy; 2 = +penoid, 3 = neovagina, 4 = breast augmentation)', 
'direction_change (direction of change in sexual orientation; 1= androphilic to gynephilic;  2=androphilic to bisexual; 3=gynephilic to androphilic; 4=gynephilic to bisexual, 5 = analloerotic to gynephilic; 6 =analloerotic to androphilic; 7 = analloerotic to bisexual; 8 = gynephilic to analloerotic) ',
'interval_horm_surg (interval from initiation of hormone therapy to sex reassignement surgery)']]
'''

''' # LOE: Notes to Self
# > Ignore NAs
#   age_of_onset -> can maybe ignore NAs because they will probably throw off predictions? or just 0 fill

# > Easier to start with
#   sex (1=MtF; 2 =FtM)
#   hormontherapy (1 =yes; 2 =no)
#   sex reassignement surgery (1= yes; 2 = no)

# > Intermediate complexity (probably good to add due to original study results; not yet sure how to handle)
#   initial_sex_orientation (1= androphilic; 2 =gynephilic; 3 = bisexual, 4 = analloerotic)

# > Should probably ignore due to high-seeming complexity
#   hormonetype
#   direction_change
'''

X = data[['sex (1=MtF; 2 =FtM)',
          'initial_sex_orientation (1= androphilic; 2 =gynephilic; 3 = bisexual, 4 = analloerotic)',
          'hormontherapy (1 =yes; 2 =no)',
          'sex reassignement surgery (1= yes; 2 = no)']]

# dependent variable
y = data['changesexorient (there has been a change in self-reported sexual orientation: 1= yes; 2 = no)']

# Clean the data
# Fill NA/empty with 0 because the numbers in the selected
# columns used in X are categorical; also, dropping NA values
# vastly reduces the number of rows from 115 to 15
#print(data) # Test print
X = X.fillna(0)         # Get X
y = y.fillna(0)         # Get y
data = data.fillna(0)   # Get anything left over
#print(data) # Test print

# LOE: Ignore NAs completely via dropping
# Also works in replacing/dropping but vastly reduces number of tuples;
# drops from 115 to 15 rows
'''
data = data.replace('NA',0)
#data = data.fillna(0)
data = data.dropna()
'''

# Post-cleaning message
print("* NA values have been successfully handled")

# refine variables after handling NA 
# LOE: not sure what you mean here?

# Perform regression 
regr = linear_model.LinearRegression()
regr.fit(X,y)

# LOE: Predict using args
# 'sex (1=MtF; 2 =FtM)', 'initial_sex_orientation (1= androphilic; 2 =gynephilic; 3 = bisexual, 4 = analloerotic)', 'hormontherapy (1 =yes; 2 =no)', 'sex reassignement surgery (1= yes; 2 = no)'
# Result should be 1 (yes) or 2 (no) or in that range
predictIfChange = regr.predict([[2,3,1,1]]) # Arbitrary for now
print(predictIfChange)

# TBA: Correlation calculations (optional)

# TBA: Z-test (compare FTM and MTF populations)