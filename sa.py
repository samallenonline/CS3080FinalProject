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

# dependent variable
y = data['changesexorient (there has been a change in self-reported sexual orientation: 1= yes; 2 = no)']

# replace NA values with a -1 to ensure consistency with categorical variables
#data[''] = data[''].replace('NA', '-1')
data['age_of_onset'] = data['age_of_onset'].replace('NA', '-1')
data['onset_before_age_of_12 (1= before or at age of 12; after age of 12)'] = data['onset_before_age_of_12 (1= before or at age of 12; after age of 12)'].replace('NA', '-1')
data['age_psychol (age of first psychological counselling)'] = data['age_psychol (age of first psychological counselling)'].replace('NA', '-1')
data['age_role (age of start everday-experience)'] = data['age_role (age of start everday-experience)'].replace('NA', '-1')
data['age_hormonetherapy (age of initiation of hormonetherapy)'] = data['age_hormonetherapy (age of initiation of hormonetherapy)'].replace('NA', '-1')
data['hormonetype (1= T transdermal; 2 = T intramuscular; 3 = E + Antiandrogen; 4 = Estradiol transdermal; 5 = Estradiol oral; 6 = estradiol +gestagen)'] = data['hormonetype (1= T transdermal; 2 = T intramuscular; 3 = E + Antiandrogen; 4 = Estradiol transdermal; 5 = Estradiol oral; 6 = estradiol +gestagen)'].replace('NA', '-1')
data['age_surgery (age of sex reassignement surgery)'] = data['age_surgery (age of sex reassignement surgery)'].replace('NA', '-1')
data['type_of_surgery (1 = hysterectomy + mastectomy; 2 = +penoid, 3 = neovagina, 4 = breast augmentation)'] = data['type_of_surgery (1 = hysterectomy + mastectomy; 2 = +penoid, 3 = neovagina, 4 = breast augmentation)'].replace('NA', '-1')
data['changesexorient (there has been a change in self-reported sexual orientation: 1= yes; 2 = no)'] = data['changesexorient (there has been a change in self-reported sexual orientation: 1= yes; 2 = no)'].replace('NA', '-1')

# Fill blank values with -1
#data[''] = data[''].replace('', '-1')
data['age_hormonetherapy (age of initiation of hormonetherapy)'] = data['age_hormonetherapy (age of initiation of hormonetherapy)'].replace('', '-1')
data['hormonetype (1= T transdermal; 2 = T intramuscular; 3 = E + Antiandrogen; 4 = Estradiol transdermal; 5 = Estradiol oral; 6 = estradiol +gestagen)'] = data['hormonetype (1= T transdermal; 2 = T intramuscular; 3 = E + Antiandrogen; 4 = Estradiol transdermal; 5 = Estradiol oral; 6 = estradiol +gestagen)'].replace('', '-1')
data['age_surgery (age of sex reassignement surgery)'] = data['age_surgery (age of sex reassignement surgery)'].replace('', '-1')
data['type_of_surgery (1 = hysterectomy + mastectomy; 2 = +penoid, 3 = neovagina, 4 = breast augmentation)'] = data['type_of_surgery (1 = hysterectomy + mastectomy; 2 = +penoid, 3 = neovagina, 4 = breast augmentation)'].replace('', '-1')

print("* NA and empty values have been successfully handled")

# refine variables after handling NA 

# perform regression 
# regr = linear_model.LinearRegression()
# regr.fit(X,y)
