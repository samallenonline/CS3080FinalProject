# CS3080 - Final Project 
# Gender transitioning and changes in self-reported sexual orientation
# Library functions version
# Sam Allen and Loe Malabanan

# import libraries 
import pandas as pd 
import numpy as np 
from statsmodels.stats.proportion import proportions_ztest # To perform z-test

# from scikit-learn 
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# For saving files/directory work
import os

print("* All libraries have been successfully imported")

##############################################################################################################################
# DATA SECTION ###############################################################################################################
##############################################################################################################################

# Load in data
data = pd.read_csv("data_simplified_preclean.csv")

# Independent variable data
X = data[['sex (1=MtF; 2 =FtM)',
          'initial_sex_orientation (1= androphilic; 2 =gynephilic; 3 = bisexual, 4 = analloerotic)',
          'hormontherapy (1 =yes; 2 =no)',
          'sex reassignement surgery (1= yes; 2 = no)']]

# Dependent variable data
y = data['changesexorient (there has been a change in self-reported sexual orientation: 1= yes; 2 = no)']

# Clean the data
# Fill NA/empty with 0 because the numbers in the selected
# columns used in X are categorical; also, dropping NA values
# vastly reduces the number of rows from 115 to 15
X = X.fillna(0)         # Get X
y = y.fillna(0)         # Get y
data = data.fillna(0)   # Get everything else

# Post-cleaning message
print("* Data and NA values have been successfully handled")
print("\n****************************************************************************************************************************\n")

##############################################################################################################################
# REGRESSION SECTION #########################################################################################################
##############################################################################################################################

# Perform regression 
regr = linear_model.LinearRegression()
regr.fit(X.values,y)

# Sample test individual data (FTM/BI/HRT/SRG)
# 'sex (1=MtF; 2 =FtM)',           'initial_sex_orientation (1= androphilic; 2 =gynephilic; 3 = bisexual, 4 = analloerotic)', 
# 'hormontherapy (1 =yes; 2 =no)', 'sex reassignement surgery (1= yes; 2 = no)'
# Result should be 1 (yes) or 2 (no) or in that range

# Labels for all
predictDataNames = [" Androphilic/Y/Y", " Androphilic/N/N",
                    " Androphilic/Y/N", " Androphilic/N/Y",
                    "  Gynephilic/Y/Y", "  Gynephilic/N/N",
                    "  Gynephilic/Y/N", "  Gynephilic/N/Y",
                    "    Bisexual/Y/Y", "    Bisexual/N/N",
                    "    Bisexual/Y/N", "    Bisexual/N/Y",
                    "Analloerotic/Y/Y", "Analloerotic/N/N",
                    "Analloerotic/Y/N", "Analloerotic/N/Y"]

# For MtF
predictDataMtFVals = [[1,1,1,1],[1,1,2,2],
                      [1,1,1,2],[1,1,2,1],
                      [1,2,1,1],[1,2,2,2],
                      [1,2,1,2],[1,2,2,1],
                      [1,3,1,1],[1,3,2,2],
                      [1,3,1,2],[1,3,2,1],
                      [1,4,1,1],[1,4,2,2],
                      [1,4,1,2],[1,4,2,1]]
predictionsMtF = [] # Save results to get means later

# For FtM
predictDataFtMVals = [[2,1,1,1],[2,1,2,2],
                      [2,1,1,2],[2,1,2,1],
                      [2,2,1,1],[2,2,2,2],
                      [2,2,1,2],[2,2,2,1],
                      [2,3,1,1],[2,3,2,2],
                      [2,3,1,2],[2,3,2,1],
                      [2,4,1,1],[2,4,2,2],
                      [2,4,1,2],[2,4,2,1]]
predictionsFtM = [] # Save results to get means later

# Start outputting results
print("Now performing linear regression to predict likeliness of self-reported change in sexuality...\nNOTE: Nearer to 1 = YES and 2 = NO.")

print("\nResults of Sample MtF Predictions (Initial Sexuality/Hormones/Surgery):")
for i in range(len(predictDataMtFVals)):
    finalPrediction = regr.predict([predictDataMtFVals[i]])
    predictionsMtF.append(finalPrediction)
    print(str(predictDataNames[i]) + ": " + str(finalPrediction))

print("\nResults of Sample FtM Predictions (Initial Sexuality/Hormones/Surgery):")
for i in range(len(predictDataFtMVals)):
    finalPrediction = regr.predict([predictDataFtMVals[i]])
    predictionsFtM.append(finalPrediction)
    print(str(predictDataNames[i]) + ": " + str(finalPrediction))

# Calculate and print all means
meanLabels = [" Androphilic: ",
              "  Gynephilic: ",
              "    Bisexual: ",
              "Analloerotic: "]
meansMtF = [np.mean(predictionsMtF[i:i+4]) for i in range(0, len(predictionsMtF), 4)]
meansFtM = [np.mean(predictionsFtM[i:i+4]) for i in range(0, len(predictionsFtM), 4)]

print("\n Overall MtF Means:")
for i in range(len(meansMtF)):
    print(meanLabels[i], meansMtF[i])

print("\n Overall FtM Means:")
for i in range(len(meansFtM)):
    print(meanLabels[i], meansFtM[i])

# Overall, androphilic FtM are most likely to report a change in
# orientation, while gynephilic MtF are second most likely to report
# a change in orientation.
# These results are consistent with this section from the study's
# results:
#   "FtM that had initially been sexually oriented towards males 
#   ( = androphilic), were significantly more likely to report on 
#   a change in sexual orientation than gynephilic, analloerotic or 
#   bisexual FtM (p  =  0.012)"

# However, our results are dubious when it comes to completely matching
# the below section from the study's results:
#   "Similarly, gynephilic MtF reported a change in sexual orientation 
#   more frequently than androphilic, analloerotic or bisexual MtF trans-
#   sexual persons (p  =  0.05)."
# In our results, gynephilic MtF on average are the second most likely to
# report a change, the first being androphilic MtF. However, our model is
# simplified in comparison to the study; we have not incorportated all of
# the data that the study used into our calculations, thus those missing
# data on other factors that could potentially influence a person's 
# orientation may account for this particular inconsistency.

##############################################################################################################################
# CORRELATION AND VISUALIZATIONS SECTION #####################################################################################
##############################################################################################################################

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
styledCorrMatrix = correlationMatrix.style.background_gradient(cmap='coolwarm') # Produces a matrix of correlations with color-coding to appear like a heatmap

# Print correlation matrix
print("\n****************************************************************************************************************************\n")
print(f"Results of correlation calculations (color-coded correlation matrix has also been exported as an HTML file):\n{correlationMatrix}")

# Export dataframe as an HTML file so it can be viewed in web browser
# File will save in CWD (where this file is run from)
# Get current working directory
dir_cwd = os.getcwd()

# Append file name to CWD path and save for use
htmlFilePath = dir_cwd + "\\output_correlationMatrix.html"

# Write to file
with open(htmlFilePath, "w") as f:
    f.write(styledCorrMatrix.to_html())

# CORRELATION RESULTS AND INTERPRETATIONS
# changesexorient and sex: 	                      0.158631
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

##############################################################################################################################
# Z-TEST SECTION #############################################################################################################
##############################################################################################################################

# Z-test will compare change in self-reported sexual orientation between FTM and MTF populations
# Proportion: percentage of the population group that reports a change in self-reported sexual orientation

# NA values (currently 0) will be removed here since they are not applicable
# Values will also be altered so that 0 = no and 1 = yes
# Make a copy of the relevant data slice to avoid SettingWithCopyWarning
ztestData = data[data['changesexorient (there has been a change in self-reported sexual orientation: 1= yes; 2 = no)'] != 0].copy()
ztestData['changesexorient (there has been a change in self-reported sexual orientation: 1= yes; 2 = no)'] = ztestData['changesexorient (there has been a change in self-reported sexual orientation: 1= yes; 2 = no)'].replace({2: 0})

# Filter data to create variables for FTM and MTF populations 
MTFData = ztestData[ztestData['sex (1=MtF; 2 =FtM)'] == 1]['changesexorient (there has been a change in self-reported sexual orientation: 1= yes; 2 = no)']
FTMData = ztestData[ztestData['sex (1=MtF; 2 =FtM)'] == 2]['changesexorient (there has been a change in self-reported sexual orientation: 1= yes; 2 = no)']

# Calculate proportions for each population group (how many 1 values under changesexorient)
MTFProp = MTFData.mean()
FTMProp = FTMData.mean()
print("\n****************************************************************************************************************************\n")
print("Percent of study participants who reported a change in sexual orientation: ")
print(f"MtF Proportion: {MTFProp:.4f}") # 0.338 - meaning 33.8% of MtF study participants reported a change in sexual orientation
print(f"FtM Proportion: {FTMProp:.4f}") # 0.222 - meaning 22.2% of FtM study participants reported a change in sexual orientation

# The above results are mostly consistent with the conclusions of the study:
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
