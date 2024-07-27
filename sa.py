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

print("\n* All libraries have been successfully imported")

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
# print(data) # Test print
X = X.fillna(0)         # Get X
y = y.fillna(0)         # Get y
data = data.fillna(0)   # Get anything left over
#print(data) # Test print

# Post-cleaning message
print("* NA values have been successfully handled")
print("\n****************************************************************************************************************************\n")

# Perform regression 
regr = linear_model.LinearRegression()
regr.fit(X.values,y)

# Arbitrary test individuals (FTM/BI/HRT/SRG)
# Sample prediction data + fitting
# 'sex (1=MtF; 2 =FtM)',           'initial_sex_orientation (1= androphilic; 2 =gynephilic; 3 = bisexual, 4 = analloerotic)', 
# 'hormontherapy (1 =yes; 2 =no)', 'sex reassignement surgery (1= yes; 2 = no)'
# Result should be 1 (yes) or 2 (no) or in that range
predictDataMtFNames = [" Androphilic/Y/Y", " Androphilic/N/N",
                       "  Gynephilic/Y/Y", "  Gynephilic/N/N",
                       "    Bisexual/Y/Y", "    Bisexual/N/N",
                       "Analloerotic/Y/Y", "Analloerotic/N/N"]
predictDataMtFVals = [[1,1,1,1],[1,1,2,2],
                      [1,2,1,1],[1,2,2,2],
                      [1,3,1,1],[1,3,2,2],
                      [1,4,1,1],[1,4,2,2]]

predictDataFtMNames = [" Androphilic/Y/Y", " Androphilic/N/N",
                       "  Gynephilic/Y/Y", "  Gynephilic/N/N",
                       "    Bisexual/Y/Y", "    Bisexual/N/N",
                       "Analloerotic/Y/Y", "Analloerotic/N/N"]
predictDataFtMVals = [[2,1,1,1],[2,1,2,2],
                      [2,2,1,1],[2,2,2,2],
                      [2,3,1,1],[2,3,2,2],
                      [2,4,1,1],[2,4,2,2]]

print("Now performing linear regression to predict likeliness of self-reported change in sexuality...\nNOTE: Nearer to 1 = YES and 2 = NO.")

print("\nResults of MtF Predictions (Initial Sexuality/Hormones/Surgery):")
for i in range(len(predictDataMtFVals)):
    finalPrediction = regr.predict([predictDataMtFVals[i]])
    print(str(predictDataMtFNames[i]) + ": " + str(finalPrediction))

print("\nResults of FtM Predictions (Initial Sexuality/Hormones/Surgery):")
for i in range(len(predictDataFtMVals)):
    finalPrediction = regr.predict([predictDataFtMVals[i]])
    print(str(predictDataFtMNames[i]) + ": " + str(finalPrediction))

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
# Documentation for corr() function -> https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html
correlationsData = data[columnsOfInterest]
correlationMatrix = correlationsData.corr()
styledCorrMatrix = correlationMatrix.style.background_gradient(cmap='coolwarm') # Produces a matrix of correlations with color-coding to appear like a heatmap

# Print correlation matrix
print("\n****************************************************************************************************************************\n")
print(f"Results of correlation calculations (color-coded correlation matrix has also been exported as an HTML file):\n{correlationMatrix}")

# Export dataframe as an HTML file so it can be viewed in web browser
# LOE: I've changed this to save in CWD (where sa.py is run from)

# Get current working directory
dir_cwd = os.getcwd()

# Append file name to CWD path and save for use
htmlFilePath = dir_cwd + "\\correlationMatrix.html"

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
# Documentation for proportions_ztest() function --> https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportions_ztest.html
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
print("\n****************************************************************************************************************************\n")
print(f"MTF Proportion: {MTFProp:.4f}") # 0.338 - meaning 33.8% of MTF study participants reported a change in sexual orientation
print(f"FTM Proportion: {FTMProp:.4f}") # 0.222 - meaning 22.2% of FTM study participants reported a change in sexual orientation

# ^^ These results are mostly consistent with the conclusions of the study:
# "About one third of MtF (32.9 %, N  =  23) reported a change in sexual orientation during 
# their life, in contrast to 22.2 % (N  =  10) in the FtM group (n.s.)."

# Count the number of successes and trials for each group
count = [sum(MTFData == 1), sum(FTMData == 1)] # Number of successes
nobs = [len(MTFData), len(FTMData)] # Number of observations 

# Perform z-test and save results to a variable 
zTestResults = proportions_ztest(count, nobs)
print("\nResults of z-test for proportions:\n", zTestResults)

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
