import pandas as pd
import Utilities as ut
import numpy as np
from copy import deepcopy


def WorksheetPreProcessing(currentSheet):
        
    # Drop the 2nd column
    currentSheet = currentSheet.drop(currentSheet.columns[1], axis=1)       # axis=1 means operation is carried out over columns
        
    # Turn all zero values into 'nan'
    currentSheet.iloc[:,0] = currentSheet.iloc[:,0].str.replace("0","nan")
    
    # Replace all cells with error message with 'nan'
    for column in currentSheet.columns.values:
        currentSheet[column] = currentSheet[column].replace(r"#N/A Invalid Field", np.nan)
        currentSheet[column] = currentSheet[column].replace(r"#N/A Invalid Security", np.nan)
        currentSheet[column] = currentSheet[column].replace(r"#N/A Requesting Data...", np.nan)
        currentSheet[column] = currentSheet[column].replace(r"—", np.nan)
        currentSheet[column] = currentSheet[column].fillna("")

    # Redefine column names
    currentSheet.columns = currentSheet.iloc[2]
    
    # Drop top 4 rows
    currentSheet = currentSheet.drop(currentSheet.index[[0,1,2,3]])
    
    # Rename the 1st column
    currentSheet.columns.values[0] = "Companies"
                    
    # Reset all the row indexes
    #currentSheet = currentSheet.reset_index(drop=True)
    currentSheet = currentSheet.set_index("Companies")
        
    #currentSheet.attributes = [str(x) for x in currentSheet.iloc[:,0]]
    
    return currentSheet
    


def calculateYearlyDiffs(financialReportMetrics_ByCompany, quarterList):
    
    # Calculate quarterly changes of all metrics, when possible
    financialReportMetrics_Diff_ByCompany = {}
    for companyName, companyMetrics in financialReportMetrics_ByCompany.items():

        companyMetrics_diff = {}
        for metricName, metricValues in companyMetrics.items():

            metricValues_diff = {}
            for idx in range(4, len(quarterList)):
                lastQuarter = quarterList[idx-4]
                currentQuarter = quarterList[idx]
                
                metricValue_lastQuarter     = metricValues[lastQuarter]
                metricValue_currentQuarter  = metricValues[currentQuarter]
            
                canFindDiff = True
                if metricValue_currentQuarter == "#N/A N/A" or metricValue_currentQuarter == "nan" or metricValue_currentQuarter == '' or metricValue_currentQuarter == '—':
                    canFindDiff = False
                    
                if metricValue_lastQuarter == "#N/A N/A" or metricValue_lastQuarter == "nan" or metricValue_lastQuarter == '' or metricValue_lastQuarter == '—':
                    canFindDiff = False
            
                goodToGo = True
                yearlyChange = float()
                if canFindDiff == True:
                    
                    '''
                    Pay attention to how ratio features and non-ratio features are treated differently
                    '''
                    
                    if metricName in ut.ratioFeatures:
                        yearlyChange = metricValue_currentQuarter - metricValue_lastQuarter
                    else:
                        if metricValue_lastQuarter != 0:
                            yearlyChange = (metricValue_currentQuarter - metricValue_lastQuarter) / metricValue_lastQuarter
                        else:
                            goodToGo = False
                    
                    if goodToGo == True:
                        metricValues_diff[currentQuarter] = yearlyChange
                
            metricName = metricName + '_Y_Change'
            companyMetrics_diff[metricName] = metricValues_diff
            
        financialReportMetrics_Diff_ByCompany[companyName] = companyMetrics_diff
    
    return financialReportMetrics_Diff_ByCompany




def calculateQuarterlyDiffs(financialReportMetrics_ByCompany, quarterList):
    
    print("start")
    
    # Calculate quarterly changes of all metrics, when possible
    financialReportMetrics_Diff_ByCompany = {}
    for companyName, companyMetrics in financialReportMetrics_ByCompany.items():

        companyMetrics_diff = {}
        for metricName, metricValues in companyMetrics.items():

            metricValues_diff = {}
            for idx in range(1, len(quarterList)):
                lastQuarter = quarterList[idx-1]
                currentQuarter = quarterList[idx]
                
                metricValue_lastQuarter     = metricValues[lastQuarter]
                metricValue_currentQuarter  = metricValues[currentQuarter]
            
                canFindDiff = True
                if metricValue_currentQuarter == "#N/A N/A" or metricValue_currentQuarter == "nan" or metricValue_currentQuarter == '' or metricValue_currentQuarter == '—':
                    canFindDiff = False
                    
                if metricValue_lastQuarter == "#N/A N/A" or metricValue_lastQuarter == "nan" or metricValue_lastQuarter == '' or metricValue_lastQuarter == '—':
                    canFindDiff = False
            
                #print("companyName = %s, metricName = %s, quarter = %s" % (companyName, metricName, currentQuarter))
                
                goodToGo = True
                quarterlyChange = float()
                if canFindDiff == True:
                    
                    '''
                    Pay attention to how ratio features and non-ratio features are treated differently
                    '''
                    
#                    if metricName == 'PS Ratios' and companyName == "ADI":
#                        print("metricName = %s, quarter = %s, current=%d, last=%d" % (metricName, currentQuarter, metricValue_currentQuarter, metricValue_lastQuarter))
                    
                    
                    if metricName in ut.ratioFeatures:
                        quarterlyChange = metricValue_currentQuarter - metricValue_lastQuarter
                    else:
                        if metricValue_lastQuarter != 0:
                            quarterlyChange = (metricValue_currentQuarter - metricValue_lastQuarter) / metricValue_lastQuarter
                        else:
                            goodToGo = False
                    
                    if goodToGo == True:
                        metricValues_diff[currentQuarter] = quarterlyChange
                
            metricName = metricName + '_Q_Change'
            companyMetrics_diff[metricName] = metricValues_diff
            
        financialReportMetrics_Diff_ByCompany[companyName] = companyMetrics_diff
           
    return financialReportMetrics_Diff_ByCompany











