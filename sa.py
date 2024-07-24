# CS3080 - Final Project 
# Gender transitioning and changes in self-reported sexual orientation

# using this tutorial to help me out --> https://www.w3schools.com/python/python_ml_multiple_regression.asp

# import libraries 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

# from scikit-learn 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

print("* All libraries have been successfull imported")

# load in data
data = pd.read_csv("sexual_orientation_auer_anonymized.csv")

# list of independent variables 
''' # Simplified below for basic testing purposes
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

X = data[['sex (1=MtF; 2 =FtM)',
'hormontherapy (1 =yes; 2 =no)',
'sex reassignement surgery (1= yes; 2 = no)',]]

# dependent variable
y = data['changesexorient (there has been a change in self-reported sexual orientation: 1= yes; 2 = no)']

# replace NA values with a -1 to ensure consistency with categorical variables
#data[''] = data[''].replace('NA', '-1')
# See: https://stackoverflow.com/questions/38117016/update-pandas-dataframe-with-str-replace-vs-replace

''' # Loe notes to self
> ignore NAs
  age_of_onset -> can ignore NAs because they will probably throw off predictions

> easier to start with
  sex (1=MtF; 2 =FtM)
  hormontherapy (1 =yes; 2 =no)
  sex reassignement surgery (1= yes; 2 = no)

> intermediate complexity
  initial_sex_orientation (1= androphilic; 2 =gynephilic; 3 = bisexual, 4 = analloerotic)

> should probably ignore due to complexity
  hormonetype
  direction_change
'''

print("* NA and empty values have been successfully handled")

# refine variables after handling NA 
# not sure what you mean here? -Loe

# perform regression 
# regr = linear_model.LinearRegression()
# regr.fit(X,y)
