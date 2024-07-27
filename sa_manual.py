# CS3080 - Final Project
# MANUAL FUNCTIONS FOR CORRELATION CALCULATIONS AND Z-TEST FOR PROPORTIONS 

# Import libraries 
import pandas as pd # For CSV handling
import math # For math calculations - specifically to calculate error function

###############################################################
# FUNCTION SECTION ############################################
###############################################################

# Functions for simple math calculations ######################
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

# Function to calculate correlations ##########################
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

###############################################################
# MAIN PROGRAM SECTION ########################################
###############################################################

# MANUAL FUNCTIONS START HERE! 
# Load in data using pandas function - get rid of quotes in column names 
df = pd.read_csv("data_simplified_preclean.csv", quotechar='"', skipinitialspace=True)
df.columns = df.columns.str.strip().str.replace("'", "")

###############################################################
# CORRELATION SECTION #########################################
###############################################################

# Print column names
print("\nColumn names for dataset:\n\n", df.columns)
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
    
###############################################################
# Z-TEST SECTION ##############################################
###############################################################

# Perform z-test and print results 
print("\n****************************************************************************************************************************\n")
print("Results of proportion calculations and z-test: \n")

# Prepare data for z-test 
# I will be removing NA values (currently 0) here since they are not applicable, and altering the values so that 0 = no and 1 = yes
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

# Count the number of successes and trials for each group
count = [sum(MTFData == 1), sum(FTMData == 1)]  # Number of successes
nobs = [len(MTFData), len(FTMData)]  # Number of observations

# MANUAL Z-TEST
pCombined = (count[0] + count[1]) / (nobs[0] + nobs[1])
stdError = ((pCombined * (1 - pCombined) * (1 / nobs[0] + 1 / nobs[1])) ** 0.5)

# Calculate z-statistic 
zStatistic = (MTFProp - FTMProp) / stdError
# Calculate p-value 
pValue = 2 * (1 - (0.5 * (1 + math.erf(zStatistic / (2 ** 0.5)))))

# Print results of z-test
print(f"Z-Statistic:    {zStatistic:.4f}")  # 1.3423
print(f"P-Value:        {pValue:.4f}")      # 0.1795

# ^^Results of the manual z-test are close but not identical to the values returned using the proportions_ztest() 
# function in the statsmodels library. The results returned by this function are as follows:
# Z-Statistic: 1.3277
# P-Value: 0.1843