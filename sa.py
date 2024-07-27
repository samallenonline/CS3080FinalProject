# CS3080 - Final Project 
# Gender transitioning and changes in self-reported sexual orientation

# using this tutorial to help me out --> https://www.w3schools.com/python/python_ml_multiple_regression.asp

# import libraries 
import pandas as pd 
import numpy as np 
# import matplotlib.pyplot as plt # Optional to use - for visualizations
# import seaborn as sns # Optional to use - for visualizations
from statsmodels.stats.proportion import proportions_ztest # To perform z-test

# from scikit-learn 
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# For saving files/directory work
import os

print("* All libraries have been successfully imported")

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

# Perform regression 
regr = linear_model.LinearRegression()
regr.fit(X,y)

# LOE: Predict using args
# 'sex (1=MtF; 2 =FtM)', 'initial_sex_orientation (1= androphilic; 2 =gynephilic; 3 = bisexual, 4 = analloerotic)', 'hormontherapy (1 =yes; 2 =no)', 'sex reassignement surgery (1= yes; 2 = no)'
# Result should be 1 (yes) or 2 (no) or in that range
# Arbitrary test individuals (FTM/BI/HRT/SRG)
predictIfChange1311 = regr.predict([[1,3,1,1]]) # MtF bi YES HRT/surgery
predictIfChange1322 = regr.predict([[1,3,2,2]]) # MtF bi NO HRT/surgery
predictIfChange1111 = regr.predict([[1,1,1,1]]) # MtF andr YES HRT/surgery
predictIfChange1122 = regr.predict([[1,1,2,2]]) # MtF andr NO HRT/surgery
predictIfChange1211 = regr.predict([[1,2,1,1]]) # MtF gyn YES HRT/surgery
predictIfChange1222 = regr.predict([[1,2,2,2]]) # MtF gyn NO HRT/surgery

predictIfChange2311 = regr.predict([[2,3,1,1]]) # FtM bi YES HRT/surgery
predictIfChange2322 = regr.predict([[2,1,2,2]]) # FtM bi NO HRT/surgery
predictIfChange2111 = regr.predict([[2,1,1,1]]) # FtM andr YES HRT/surgery
predictIfChange2122 = regr.predict([[2,1,2,2]]) # FtM andr NO HRT/surgery
predictIfChange2211 = regr.predict([[2,2,1,1]]) # FtM gyn YES HRT/surgery
predictIfChange2222 = regr.predict([[2,2,2,2]]) # FtM gyn NO HRT/surgery
print("Predictions for if a person will report a change under certain conditions: ")
print("Person 1311: " + str(predictIfChange1311))
print("Person 1322: " + str(predictIfChange1322))
print("Person 1111: " + str(predictIfChange1111))
print("Person 1122: " + str(predictIfChange1122))
print("Person 1211: " + str(predictIfChange1211))
print("Person 1222: " + str(predictIfChange1222))
print() # Separator
print("Person 2311: " + str(predictIfChange2311))
print("Person 2322: " + str(predictIfChange2322))
print("Person 2111: " + str(predictIfChange2111))
print("Person 2122: " + str(predictIfChange2122))
print("Person 2211: " + str(predictIfChange2211))
print("Person 2222: " + str(predictIfChange2222))

# SAM: Correlation calculations and visualizations 
# Selecting columns to be used for correlations matrix 
columnsOfInterest = [
    'sex (1=MtF; 2 =FtM)',
    'initial_sex_orientation (1= androphilic; 2 =gynephilic; 3 = bisexual, 4 = analloerotic)',
    'hormontherapy (1 =yes; 2 =no)',
    'sex reassignement surgery (1= yes; 2 = no)',
    'changesexorient (there has been a change in self-reported sexual orientation: 1= yes; 2 = no)'
]

# Utilizing corr() function from pandas library to calculate correlations 
correlationsData = data[columnsOfInterest]
correlationMatrix = correlationsData.corr()
# UNCOMMENT LATER
#print("Correlation matrix: \n" + str(correlationMatrix))
styledCorrMatrix = correlationMatrix.style.background_gradient(cmap='coolwarm') # Produces a matrix of correlations with color-coding to appear like a heatmap

# Export dataframe as an HTML file so it can be viewed in web browser
# LOE: I've changed this to save in CWD (where sa.py is run from)

# Get current working directory
dir_cwd = os.getcwd()

# Append file name to CWD path and save for use
htmlFilePath = dir_cwd + "\\correlationMatrix2.html"

# Write to file
with open(htmlFilePath, "w") as f:
    f.write(styledCorrMatrix.to_html())

# CORRELATION RESULTS AND INTERPRETATIONS
# changesexorient and sex: 	                        0.158631
# changesexorient and initial_sex_orientation:      0.123523
# changesexorient and hormontherapy:                0.074691
# cchangesexorient and sex_reassignment_surgery:    0.155776

# According to the results of the correlation calculations, there appears to be very weak correlations 
# between change in self-reported sexual orientation and initial sex, intial sex orientation, whether the 
# participant is taking hormone therapy, and whether the participant has undergone gender-affirming surgery. 

# ^^ These correlations are consistent with the conclusions of the study "Transgender Transitioning and Change 
# of Self-Reported Sexual Orientation", which states, "...self-reported change in sexual orientation is a common 
# phenomenon in transsexual persons. Transition was not directly involved in this change, since a significant 
# number of participants reported a change in sexual orientation prior to first psychological counseling and 
# prior to initiation of cross-sex hormone treatment."

# SAM: Z-test (compare change in self-reported sexual orientation between FTM and MTF populations) 
# Reference for the proportions_ztest function --> https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportions_ztest.html
# Proportion: percentage of the population group that reports a change in self-reported sexual orientation

# I will be removing NA values (currently 0) here since they are not applicable, and altering the values so that 0 = no and 1 = yes
ztestData = data[data['changesexorient (there has been a change in self-reported sexual orientation: 1= yes; 2 = no)'] != 0]
ztestData['changesexorient (there has been a change in self-reported sexual orientation: 1= yes; 2 = no)'] = ztestData['changesexorient (there has been a change in self-reported sexual orientation: 1= yes; 2 = no)'].replace({2: 0})

# Filter data to create variables for FTM and MTF populations 
MTFData = ztestData[ztestData['sex (1=MtF; 2 =FtM)'] == 1]['changesexorient (there has been a change in self-reported sexual orientation: 1= yes; 2 = no)']
FTMData = ztestData[ztestData['sex (1=MtF; 2 =FtM)'] == 2]['changesexorient (there has been a change in self-reported sexual orientation: 1= yes; 2 = no)']

# Calculate proportions for each population group (how many 1 values under changesexorient)
MTFProp = MTFData.mean()
FTMProp = FTMData.mean()
print("Percent of MTF participants who reported a change in sexual orientation: " + str(MTFProp * 100)) # 0.338 - meaning 33.8% of MTF study participants reported a change in sexual orientation
print("Percent of FTM participants who reported a change in sexual orientation: " + str(FTMProp * 100)) # 0.222 - meaning 22.2% of FTM study participants reported a change in sexual orientation

# ^^ These results are mostly consistent with the conclusions of the study:
# "About one third of MtF (32.9 %, N  =  23) reported a change in sexual orientation during 
# their life, in contrast to 22.2 % (N  =  10) in the FtM group (n.s.)."

# Count the number of successes and trials for each group
count = [sum(MTFData == 1), sum(FTMData == 1)] # Number of successes
nobs = [len(MTFData), len(FTMData)] # Number of observations 

# Perform z-test and save results to a variable 
zTestResults = proportions_ztest(count, nobs)
print("Z-test results (test statistic, P-value): " + str(zTestResults))

# Z-TEST RESULTS
# Test statistic: 1.33
# P-value: 0.18

# Null hypothesis: There is no significant difference in the frequency of change in self-reported sexual orientation
# between FTM and MTF population groups 
# Alternative hypothesis: There is a significant difference in the frequency of change in self-reported sexual orientation
# between FTM and MTF population groups 

# ^^Considering the p-value of the z-test, the result is not significant and the null hypothesis is 
# accepted. Therefore, there is no significant difference in the frequency of change in self-reported 
# sexual orientation between FTM and MTF population groups.