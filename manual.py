import csv
import os

# Get current working directory ##############################
dir_cwd = os.getcwd()

# Append file name to CWD path and save for use ##############
path_dataCSV = dir_cwd + "\\data_simplified_preclean.csv"

# Read CSV to get data #######################################
data = []
with open(path_dataCSV, 'r') as file_origFile:
    # Make csv reader object
    csvreader = csv.reader(file_origFile)
    
    # Extract fields names in first tow
    var_fields = next(csvreader)

    # Extract data from remaining rows
    for row in csvreader:
        data.append(row)

# Clean data; fill NAs/empty with 0s #########################
for i in range(len(data)):
    for j in range(len(data[i])):
        if (data[i][j] == '' or data[i][j] == 'NA'):
            data[i][j] = 0

print(data)

# Separate X and y ###########################################
var_xCols = [1,2,7,10]
X = []
y = []

# Extract from data; X is col 1,2,7,10 #######################
for col in var_xCols:
    var_newRow = []

    for row in data:
        var_newRow.append(int(row[col]))
    
    X.append(var_newRow)

# Test print
print("X values:\n" + str(X))

# Extract from data; y is only col 13 ########################
for row in data:
    y.append(int(row[13]))

# Test print
print("y values:\n" + str(y))

# Get row means in X #########################################
Xmeans = []
for row in X:
    var_rowSum = sum(row)
    var_rowMean = var_rowSum / len(row)
    Xmeans.append(var_rowMean)

# Test print
print("Means of rows in X: " + str(Xmeans))

# Get y mean #################################################
var_yMean = sum(y) / len(y)
print("Mean of y: " + str(var_yMean))

# Fit function equivalent ####################################
#def ourFit(X,y):