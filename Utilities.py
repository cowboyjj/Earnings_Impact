import numpy as np

MACHINE_NAME = "zhengxin"
#MACHINE_NAME = "lenovo"

def GetStockPriceDataDir():
    return r"C:\Users" + "\\" + MACHINE_NAME + r'\Dropbox\Imperial College\Project - Earnings Impact\Data\Stock Prices'

def GetReportDataDir():
    return r"C:\Users" + "\\" + MACHINE_NAME + r'\Dropbox\Imperial College\Project - Earnings Impact\Data\Financial Reports'

def GetReportDataDir2():
    return r"C:\Users" + "\\" + MACHINE_NAME + r'\Dropbox\Imperial College\Project - Earnings Impact\Data\Financial Reports 2'

def GetVolumeDataDir():
    return r"C:\Users" + "\\" + MACHINE_NAME + r'\Dropbox\Imperial College\Project - Earnings Impact\Data\Volume Data'

def GetTechnicalDataDir():
    return r"C:\Users" + "\\" + MACHINE_NAME + r'\Dropbox\Imperial College\Project - Earnings Impact\Data\Technical Indicator Data'

def GetMacroDataDir():
    return r"C:\Users" + "\\" + MACHINE_NAME + r'\Dropbox\Imperial College\Project - Earnings Impact\Data\Macro Data'

def GetSectorDataDir():
    return r"C:\Users" + "\\" + MACHINE_NAME + r'\Dropbox\Imperial College\Project - Earnings Impact\Data\Sector Information'

def GetEarningsDataDir():
    return r"C:\Users" + "\\" + MACHINE_NAME + r'\Dropbox\Imperial College\Project - Earnings Impact\Data\Earnings Data'

def GetFinancialReportMetricsDir():
    return r"C:\Users" + "\\" + MACHINE_NAME + r'\Dropbox\Imperial College\Project - Earnings Impact\Data\Financial Report Metrics'

def GetShortInterestRatioDir():
    return r"C:\Users" + "\\" + MACHINE_NAME + r'\Dropbox\Imperial College\Project - Earnings Impact\Data\Short Interests'

def GetReleaseDateTimeDir():
    return r"C:\Users" + "\\" + MACHINE_NAME + r'\Dropbox\Imperial College\Project - Earnings Impact\Data\Release Date Time'

def GetMarketCapDir():
    return r"C:\Users" + "\\" + MACHINE_NAME + r'\Dropbox\Imperial College\Project - Earnings Impact\Data\Market Cap'

ratioFeatures  = ['Current Ratio', \
                  'Dividend Payout Ratio', \
                  'Dividend Yield', \
                  'Inventory Turnover', \
                  'Net Debt to EBIT', \
                  'Operating Margin', \
                  'PB Ratios', \
                  'PC Ratios', \
                  'PE Ratios', \
                  'PS Ratios', \
                  'Quick Ratio', \
                  'Return On Assets', \
                  'Return On Common Equity', \
                  'Total Debt to Total Assets', \
                  'Total Debt to Total Equity']

def WriteAttributeQuarterDataToCSV(attributes):
    with open("mycsv2.csv","w") as f:
        for quarter in attributes.items():
            quarterName = quarter[0]
            f.writelines('\n')
            f.writelines(quarterName)
            f.writelines('\n')
            f.writelines('\n')
            
            companiesInQuarter = quarter[1]
            for companies in companiesInQuarter.items():
                companyName = companies[0]
                companyData = companies[1]
                
                f.writelines(companyName)
                f.writelines(',')
                
                for feature in companyData:
                    f.writelines(feature)
                    f.writelines(',')    
                
                f.writelines('\n')
                


def trimEmptyRows(sheet, threshold = 12):
    
    # Reset all row indexex
    #
    # Important Note:
    # This step is extremely important. The row dropping line below is based on row indexes. Should
    # row indexes get mixed up, rows can be unintentionally dropped !!
    sheet = sheet.reset_index(drop=True)
    
    # Get dataframe dimensions
    rowCount = sheet.shape[0]
    columnCount = sheet.shape[1]
    
    # Count the number of cells with valid data, per rows
    countNaN_perRow = sheet.isnull().sum(axis=1)              # axis=1 means operation is carried out over columns
    countDash_perRow = (sheet == r'—').sum(axis=1)
    countEmpty_perRow = (sheet == '').sum(axis=1)
    countZero_perRow = (sheet == 0).sum(axis=1)
    countValidData_perRow = [columnCount - (x + y + z + w) for x,y,z,w in zip (countNaN_perRow, countDash_perRow, countEmpty_perRow, countZero_perRow) ]
    
    # Remove gap rows and rows with no data at all
    dropRowIndex = [i for i in range(rowCount) if countValidData_perRow[i] < threshold]
    sheet = sheet.drop(sheet.index[ dropRowIndex ], axis=0)
    
    # Reset all row indexex
    sheet = sheet.reset_index(drop=True)
    return sheet

    
def trimColumns(sheet, threshold):
    # Get dataframe dimensions
    rowCount = sheet.shape[0]
    columnCount = sheet.shape[1]
    
    # Count the number of cells with valid data, per columns
    countNaN_perColumn = (sheet == 'nan').sum(axis=0)
    countNull_perColumn = sheet.isnull().sum(axis=0)           # axis=0 means operation is carried out over rows
    countDash_perColumn = (sheet == r'—').sum(axis=0)
    countEmpty_perColumn = (sheet == '').sum(axis=0)
    countZero_perColumn = (sheet == 0).sum(axis=0)
    
    countValidData_perColumn = [rowCount - (x + y + z + u + w) for x,y,z,u,w in zip(countNaN_perColumn, countNull_perColumn, countDash_perColumn, countEmpty_perColumn, countZero_perColumn) ]
    # Remove columns with no data at all
    sheet = sheet.drop(sheet.columns[ [j for j in range(columnCount) if countValidData_perColumn[j] < threshold] ], axis=1)
    
    # Reset all row indexex
    #sheet = sheet.reset_index(drop=True)
    return sheet


def filterOutliers(method, inputPoints, outputPointsRegress, outputPointsClass, trainingPointNames):
    inputs = []
    outputsRegress = []
    outputsClass = []
    names = []
    
    if method == 'quantile':
        from copy import deepcopy
        temp = deepcopy(outputPointsRegress)
        sortedOutputs = sorted(temp)
        q1, q3= np.percentile(sortedOutputs,[25,75])
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        for idx in range(len(outputPointsRegress)):
            y = outputPointsRegress[idx]
            if y <= upper_bound and y >= lower_bound:
                inputs.append(inputPoints[idx])
                outputsRegress.append(outputPointsRegress[idx])
                outputsClass.append(outputPointsClass[idx])
                names.append(trainingPointNames[idx])
                
    elif method == 'zscore':
        threshold = 3
        mean = np.mean(outputPointsRegress)
        std = np.std(outputPointsRegress)
        for idx in range(len(outputPointsRegress)):
            y = outputPointsRegress[idx]
            zscore = (y-mean)/std
            if zscore <= threshold and zscore >= -1*threshold:
                inputs.append(inputPoints[idx])
                outputsRegress.append(outputPointsRegress[idx])
                outputsClass.append(outputPointsClass[idx])
                names.append(trainingPointNames[idx])
    else:
        inputs = inputPoints
        outputsRegress = outputPointsRegress
        outputsClass = outputPointsClass
        names = trainingPointNames
        
    return (inputs, outputsRegress, outputsClass, names)
        

def transformQuarterName(quarterName):
    newQuarterName = quarterName[4:8] + ' ' + quarterName[0:3]
    return newQuarterName


from sklearn.utils.class_weight import compute_class_weight

def computeSampleWeights(data_to_be_weighted, TestType, doSampleWeight):
    sample_weight = np.ones(len(data_to_be_weighted))
    if doSampleWeight == True:
        if TestType != "Regression" or TestType == "Classification":
            #classCount = len(np.unique(data_to_be_weighted))
            #class_weight = compute_class_weight('balanced', np.unique(y_train.T.values[0]) ,y_train.T.values[0])
            class_weight = compute_class_weight('balanced', np.unique(data_to_be_weighted.T.values) ,data_to_be_weighted.T.values)
                    
            weightbase = class_weight[0]
            for index, w in np.ndenumerate(class_weight):
                class_weight[index] = w/weightbase
                   
            for index, y in np.ndenumerate(data_to_be_weighted.T.values):    
                sample_weight[index] = class_weight[int(y)]
    return sample_weight