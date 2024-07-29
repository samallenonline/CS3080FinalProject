# CS3080 - Final Project
# MANUAL FUNCTIONS FOR CORRELATION CALCULATIONS AND Z-TEST FOR PROPORTIONS 

# Import libraries 
import pandas as pd # For CSV handling
import math # For math calculations - specifically to calculate error function
import numpy as np # For more math in regression section

##############################################################################################################################
# FUNCTION SECTION ###########################################################################################################
##############################################################################################################################

# Functions for regression ###################################################################################################
def addBiasToX(npX):
    bias = np.ones((len(npX),1))
    X = np.append(bias, npX, axis=1)

    return X

def fitXdata(inX):
    # Convert to numpy float array
    X = np.array(inX)
    X = X.astype(float)

    # Add bias as is necessary for normal equation method
    X = addBiasToX(X)

    return X

def fityData(iny):
    # Convert to numpy float array and reshape
    y = np.array(iny)
    y = y.astype(float)
    y = np.array(y).reshape((len(y),1))

    return y

def fitPredictionData(inData):
    inData = np.insert(inData, 0, 1, axis=0)
    return inData

def getNormalBeta(npX,npy):
    # Use normal equation beta = [(X^T X)^(-1)] X^T y
    beta = np.dot((np.linalg.inv(np.dot(npX.T,npX))), np.dot(npX.T,npy))
    return beta

# Functions for math calculations 

def getPrediction(npX,beta):
    return np.dot(npX,beta)

# Functions for simple math calculations #####################################################################################
def calculateMean(nums):
    return sum(nums) / len(nums)
    
def calculateStddev(nums):
    mu = calculateMean(nums)
    return (sum((x - mu) ** 2 for x in nums) / (len(nums) - 1)) ** 0.5
    
def calculatePearsonCorr(x, y):
    muX, muY = calculateMean(x), calculateMean(y)
    stddevX, stddevY = calculateStddev(x), calculateStddev(y)
    covariance = sum((xi - muX) * (yi - muY) for xi, yi in zip(x, y))
    return covariance / ((len(x) - 1) * stddevX * stddevY)

# Function to calculate correlations #########################################################################################
def calculateCorrelations(dataFrame):    
    # Initialize variables 
    correlations = {}
    columns = dataFrame.columns
    numColumns = len(columns)

    # Iterate through variables to calculate Pearson correlation coefficients 
    for i in range(numColumns):
        for j in range(i, numColumns):
            # Omit any NA values 
            col1 = dataFrame.iloc[:, i].dropna()
            col2 = dataFrame.iloc[:, j].dropna()

            if not col1.empty and not col2.empty:

                # Check if both columns have the same length 
                common_indices = col1.index.intersection(col2.index)
                col1 = col1.loc[common_indices]
                col2 = col2.loc[common_indices]

                if not col1.empty and not col2.empty:
                    corr = calculatePearsonCorr(col1, col2)
                    correlations[(columns[i], columns[j])] = corr
                    correlations[(columns[j], columns[i])] = corr 

    return correlations

##############################################################################################################################
# MAIN PROGRAM SECTION #######################################################################################################
##############################################################################################################################

# MANUAL FUNCTIONS START HERE! 
# Load in data using pandas function - get rid of quotes in column names 
df = pd.read_csv("data_simplified_preclean.csv", quotechar='"', skipinitialspace=True)
df.columns = df.columns.str.strip().str.replace("'", "")

##############################################################################################################################
# REGRESSION SECTION #########################################################################################################
##############################################################################################################################

# Independent variables data
Xdata = df[['sex (1=MtF; 2 =FtM)',
          'initial_sex_orientation (1= androphilic; 2 =gynephilic; 3 = bisexual, 4 = analloerotic)',
          'hormontherapy (1 =yes; 2 =no)',
          'sex reassignement surgery (1= yes; 2 = no)']]

# Dependent variable data
ydata = df['changesexorient (there has been a change in self-reported sexual orientation: 1= yes; 2 = no)']

# Clean the data
# Fill NA/empty with 0 because the numbers in the selected
# columns used in X are categorical; also, dropping NA values
# vastly reduces the number of rows from 115 to 15
Xdata = Xdata.fillna(0)
ydata = ydata.fillna(0)

# Fit X and y for calculation purposes
Xdata = fitXdata(Xdata)
ydata = fityData(ydata)

# Use normal equation method to get beta
beta = getNormalBeta(Xdata,ydata)

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

# Start outputting results
print("Now performing linear regression to predict likeliness of self-reported change in sexuality...\nNOTE: Nearer to 1 = YES and 2 = NO.")

print("\nResults of Sample MtF Predictions (Initial Sexuality/Hormones/Surgery):")
for i in range(len(predictDataMtFVals)):
    predictData = fitPredictionData(predictDataMtFVals[i])
    finalPrediction = getPrediction(predictData,beta)
    print(str(predictDataMtFNames[i]) + ": " + str(finalPrediction))

print("\nResults of Sample FtM Predictions (Initial Sexuality/Hormones/Surgery):")
for i in range(len(predictDataFtMVals)):
    predictData = fitPredictionData(predictDataFtMVals[i])
    finalPrediction = getPrediction(predictData,beta)
    print(str(predictDataFtMNames[i]) + ": " + str(finalPrediction))

# Separator
print("\n****************************************************************************************************************************\n")

##############################################################################################################################
# CORRELATION SECTION ########################################################################################################
##############################################################################################################################

# Print column names
print("Column names for dataset:\n\n", df.columns)
print("\n****************************************************************************************************************************\n")

# Selecting columns to be used to calculate correlations
columnsOfInterest = [
    'sex (1=MtF; 2 =FtM)',
    'initial_sex_orientation (1= androphilic; 2 =gynephilic; 3 = bisexual, 4 = analloerotic)',
    'hormontherapy (1 =yes; 2 =no)',
    'sex reassignement surgery (1= yes; 2 = no)',
    'changesexorient (there has been a change in self-reported sexual orientation: 1= yes; 2 = no)'
]

# Convert columns of interest into a dataframe and assign to variable 
filteredData = df[columnsOfInterest]

# Calculate correlations using custom function 
correlations = calculateCorrelations(filteredData)

# Print correlations for each pair of variables 
print("Results of correlation calculations: \n") 
for (col1, col2), corr in correlations.items():
    print(f"Correlation between {col1} and {col2}: {corr:.4f}")

#^^Results of the manual correlation calculations are identical to the values returned using the corr() function
# in the pandas library.
    
##############################################################################################################################
# Z-TEST SECTION #############################################################################################################
##############################################################################################################################

# Perform z-test and print results 
print("\n****************************************************************************************************************************\n")
print("Results of proportion calculations and z-test: \n")

# Prepare data for z-test 
# We will be removing NA values (currently 0) here since they are not applicable, and altering the values so that 0 = no and 1 = yes
ztestData = df[df['changesexorient (there has been a change in self-reported sexual orientation: 1= yes; 2 = no)'] != 0]
ztestData['changesexorient (there has been a change in self-reported sexual orientation: 1= yes; 2 = no)'] = ztestData['changesexorient (there has been a change in self-reported sexual orientation: 1= yes; 2 = no)'].replace({2: 0})

# Filter data to create variables for FTM and MTF populations 
MTFData = ztestData[ztestData['sex (1=MtF; 2 =FtM)'] == 1]['changesexorient (there has been a change in self-reported sexual orientation: 1= yes; 2 = no)']
FTMData = ztestData[ztestData['sex (1=MtF; 2 =FtM)'] == 2]['changesexorient (there has been a change in self-reported sexual orientation: 1= yes; 2 = no)']

# Calculate proportions for each population group (how many 1 values under changesexorient)
MTFProp = MTFData.mean()
FTMProp = FTMData.mean()
print(f"MTF Proportion: {MTFProp:.4f}") # 0.3382
print(f"FTM Proportion: {FTMProp:.4f}") # 0.2222

# Count the number of successes and trials for each group and convert to numpy arrays
count = np.array([sum(MTFData == 1), sum(FTMData == 1)])  # Number of successes
nobs = np.array([len(MTFData), len(FTMData)]) # Number of observations

# MANUAL Z-TEST
# Calculate combined proportion
pPooled = np.sum(count) * 1. / np.sum(nobs)  
nobsFact = np.sum(1. / nobs)
var = pPooled * (1 - pPooled) * nobsFact

stdDev = np.sqrt(var)  # Calculate standard deviation
zStatistic = (MTFProp - FTMProp) / stdDev # Calculate z-statistic 
pValue = 2 * (1 - 0.5 * (1 + math.erf(abs(zStatistic) / math.sqrt(2))))  # Calculate two-sided p-value

# Print results of z-test
print(f"Z-Statistic:    {zStatistic:.4f}")  # 1.3423
print(f"P-Value:        {pValue:.4f}")      # 0.1795

# ^^Results of the manual z-test are close but not identical to the values returned using the proportions_ztest() 
# function in the statsmodels library. The results returned by this function are as follows:
# Z-Statistic: 1.3277
# P-Value: 0.1843