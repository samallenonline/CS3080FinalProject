import csv
import os

# Get current working directory
dir_cwd = os.getcwd()

# Append file name to CWD path and save for use
path_dataCSV = dir_cwd + "\\data_simplified_preclean.csv"

# Read CSV to get data
data = []
with open(path_dataCSV, 'r') as file_origFile:
    # Read and store file contents ################
    # Make csv reader object
    csvreader = csv.reader(file_origFile)
    
    # Extract fields names in first tow
    var_fields = next(csvreader)

    # Extract data from remaining rows
    for row in csvreader:
        data.append(row)

#print(data)

# Fill NAs/empty with 0s
for i in range(len(data)):
    for j in range(len(data[i])):
        if (data[i][j] == '' or data[i][j] == 'NA'):
            data[i][j] = 0

print(data)