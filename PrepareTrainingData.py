import pandas as pd
import ProcessFinancialReports as reports
import time
import datetime
import math
import Utilities as ut
import numbers
import numpy as np
from copy import deepcopy
import scipy.stats
#from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

def CalculateStockPriceChangeRatios(priceData_From, priceData_To, postReleasePriceChanges_DictForm, postReleasePriceChanges_TupleForm):
    
    for company, companyAllQuarterData_From in priceData_From.items():
        
        companyData = {}
        if company in priceData_To:
            companyAllQuarterData_To = priceData_To[company]
            for quarter, quarterData_From in companyAllQuarterData_From.items():
                if quarter in companyAllQuarterData_To.keys():
                    quarterData_To = companyAllQuarterData_To[quarter]
                    ratio = (quarterData_To - quarterData_From) / quarterData_From                                          
                    companyData[quarter] = ratio
                    
                    quarter = ut.transformQuarterName(quarter)
                    postReleasePriceChanges_TupleForm[(company, quarter)] = ratio
        
        postReleasePriceChanges_DictForm[company] = companyData
        
        
        
        
def GetALLStockPriceData(scenario, priceReactions, priceReactionForward, doCAR, includeMinus1Dto1D):

    print("--- Start GetALLStockPriceData ---")
    
#    priceReaction_minus1D_to_0D = {}
#    priceReaction_0D_to_1D = {}
    priceReaction_minus1D_to_1D = {}
    priceReactionSoFar = {}
    SnP500priceReaction_minus1D_to_1D = {}
#    SnP500priceReaction_minus1D_to_0D = {}
#    SnP500priceReaction_0D_to_1D = {}
    SnP500priceReactionSoFar = {}
    
    startDate = scenario["start"]
    endDate = scenario ["end"]
    
    allStockPrices_TupleFormat = {}
    allStockPrices_DictFormat = {}
    SnP500Prices = {}
    
    start_time = time.time()
    commonKeys = set()
    
    DataDir = ut.GetStockPriceDataDir()
            
    '''
    2. Parse the second version of the price workbooks
    '''
    Stock_Price_Workbooks2 = ["Historical Stock Prices_-1D.xlsm",
                              "Historical Stock Prices_1D.xlsm",
                              "Historical Stock Prices_2D.xlsm",
                              "Historical Stock Prices_3D.xlsm",
                              "Historical Stock Prices_4D.xlsm",
                              "Historical Stock Prices_5D.xlsm",
                              "Historical Stock Prices_6D.xlsm",
                              "Historical Stock Prices_7D.xlsm",
                              "Historical Stock Prices_8D.xlsm",
                              "Historical Stock Prices_9D.xlsm",
                              "Historical Stock Prices_10D.xlsm",
                              "Historical Stock Prices_30D.xlsm",
                              "Historical Stock Prices_60D.xlsm"]
    Stock_Price_DateScenarios2 = ['-1D', '1D', '2D', '3D', '4D', '5D', '6D', '7D', '8D', '9D', '10D', '30D', '60D']
    
    # Parse the workbooks
    allBookData = []
    for book in Stock_Price_Workbooks2:
        fullWorkbookDir = DataDir + '\\' + book
        bookData = pd.ExcelFile(fullWorkbookDir)
        allBookData.append(bookData)
    
    '''
    === Advanced Warning ===
    Cell (0,0) of the spreadsheets must not have any value
    Otherwise the sheet structure will change
    === Advanced Warning ===
    '''
    
    # Go through each date scenario and extract price data
    for idx in range(len(Stock_Price_DateScenarios2)):
        scenarioName = Stock_Price_DateScenarios2[idx]
        bookData = allBookData[idx]
        
        allCompanyPrices_currentScenario = {}
        companyData_AllQuarters = {}
        
        sheetData = bookData.parse('static')    # the sheet name is 'static'
        sheetData.columns = sheetData.iloc[0]           # Use the company names as column names
        
        sheetData = ut.trimColumns(sheetData, 5)     # Get rid of empty columns. Threshold: 5 valid entries
        
        quarterNames = sheetData.index.tolist()
        quarterNames.pop(0)
        companyNames = sheetData.columns.tolist()
                    
        for companyName in companyNames:                
            quarterData = {}
            for quarterName in quarterNames:
                
                #print("quarterName=%s, companyName=%s, scenarioName=%s" % (quarterName, companyName, scenarioName))
                
                value = sheetData.loc[quarterName][companyName]
                quarterName = ut.transformQuarterName(quarterName)
                
                if isinstance(value, numbers.Number):
                    if value != "#N/A N/A" and value != "nan" and not pd.isna(value) and not pd.isnull(value) :
                        key = (companyName, quarterName)    
                        commonKeys.add(key)
                        allCompanyPrices_currentScenario[key] = value
                        quarterData[quarterName] = value
            
            companyData_AllQuarters[companyName] = quarterData
        
        allStockPrices_TupleFormat[scenarioName] = allCompanyPrices_currentScenario
        allStockPrices_DictFormat[scenarioName] = companyData_AllQuarters
    
    '''
    3. Parse S&P500 price workbooks
    '''
    Stock_Price_Workbooks3 = ["SnP500_-1D.xlsm", 
                              "SnP500_1D.xlsm", 
                              "SnP500_2D.xlsm", 
                              "SnP500_3D.xlsm", 
                              "SnP500_4D.xlsm", 
                              "SnP500_5D.xlsm", 
                              "SnP500_6D.xlsm", 
                              "SnP500_7D.xlsm", 
                              "SnP500_8D.xlsm", 
                              "SnP500_9D.xlsm", 
                              "SnP500_10D.xlsm", 
                              "SnP500_30D.xlsm",
                              "SnP500_60D.xlsm"]
    Stock_Price_DateScenarios3 = ['-1D', '1D', '2D', '3D', '4D', '5D', '6D', '7D', '8D', '9D', '10D', '30D', '60D']
    
    # Parse the workbooks
    allBookData = []
    for book in Stock_Price_Workbooks3:
        fullWorkbookDir = DataDir + '\\' + book
        bookData = pd.ExcelFile(fullWorkbookDir)
        allBookData.append(bookData)
    
    # Go through each date scenario and extract price data    
    for idx in range(len(Stock_Price_DateScenarios3)):
        scenarioName = Stock_Price_DateScenarios3[idx]
        bookData = allBookData[idx]
        
        allCompanyPrices_currentScenario = {}
        
        sheetData = bookData.parse('static')    # the sheet name is 'static'
        sheetData = ut.trimColumns(sheetData, 5)     # Get rid of empty columns. Threshold: 5 valid entries
        sheetData.columns = sheetData.iloc[0]           # Use the company names as column names
        
        quarterNames = sheetData.index.tolist()
        quarterNames.pop(0)
        companyNames = sheetData.columns.tolist()
                    
        for companyName in companyNames:                
            quarterData = {}
            for quarterName in quarterNames:
               # print("quarter=%s, company=%s,scenario=%s" % (quarterName, companyName, scenarioName))
                value = sheetData.loc[quarterName][companyName]
                quarterName = ut.transformQuarterName(quarterName)
                
                if isinstance(value, numbers.Number):
                    if value != "#N/A N/A" and value != "nan" and not pd.isna(value) and not pd.isnull(value) :
                        key = (companyName, quarterName)    
                        commonKeys.add(key)
                        allCompanyPrices_currentScenario[key] = value
                        quarterData[quarterName] = value
        
        SnP500Prices[scenarioName] = allCompanyPrices_currentScenario

    '''
    4. Calculate the daily price %change of both company stocks and S&P500 and then the alpha return
    '''
    stocks_minus1D = allStockPrices_TupleFormat["-1D"]
#    stocks_0D = allStockPrices_TupleFormat["0D"]
    stocks_1D = allStockPrices_TupleFormat["1D"]
    stocks_startDate = allStockPrices_TupleFormat[startDate]
    stocks_endDate = allStockPrices_TupleFormat[endDate]

#    snp500_all_minus20D = SnP500Prices["-20D"]    
#    snp500_all_minus19D = SnP500Prices["-19D"]
#    snp500_all_minus18D = SnP500Prices["-18D"]
#    snp500_all_minus17D = SnP500Prices["-17D"]
#    snp500_all_minus16D = SnP500Prices["-16D"]
#    snp500_all_minus15D = SnP500Prices["-15D"]
        
    snp500_all_minus1D = SnP500Prices["-1D"]
#    snp500_all_0D = SnP500Prices["0D"]
    snp500_all_1D = SnP500Prices["1D"]    
    snp500_all_startDate = SnP500Prices[startDate]
    snp500_all_endDate = SnP500Prices[endDate]
    
#    SnP500PriceChange20Day = {}
    
    for key in commonKeys:
        #if key in stocks_startDate and key in stocks_endDate and key in snp500_all_startDate and key in snp500_all_endDate and key in stocks_minus1D and key in snp500_all_minus1D and key in snp500_all_minus16D and key in snp500_all_minus17D and key in snp500_all_minus18D and key in snp500_all_minus19D and key in snp500_all_minus20D:
        if key in stocks_startDate and key in stocks_endDate and key in snp500_all_startDate and key in snp500_all_endDate and key in stocks_1D and key in snp500_all_minus1D and key in stocks_minus1D and key in snp500_all_minus1D:
            

            
#            stock_0D = stocks_0D[key]
            stock_minus1D = stocks_minus1D[key]
            stock_1D = stocks_1D[key]
#            snp500_0D = snp500_all_0D[key]
            snp500_minus1D = snp500_all_minus1D[key]
            snp500_1D = snp500_all_1D[key]
            
            if includeMinus1Dto1D == True:
                priceReaction_minus1D_to_1D[key] = (stock_1D - stock_minus1D) / stock_minus1D - (snp500_1D - snp500_minus1D) / snp500_minus1D
 #           priceReaction_minus1D_to_0D[key] = (stock_0D - stock_minus1D) / stock_minus1D
 #           priceReaction_0D_to_1D[key] = (stock_1D - stock_0D) / stock_0D
 #           priceReaction_minus1D_to_0D[key] = (stock_0D - stock_minus1D) / stock_minus1D - (snp500_0D - snp500_minus1D) / snp500_minus1D
 #           priceReaction_0D_to_1D[key] = (stock_1D - stock_0D) / stock_0D - (snp500_1D - snp500_0D) / snp500_0D
 #           SnP500priceReaction_minus1D_to_0D[key] = (snp500_0D - snp500_minus1D) / snp500_minus1D
 #           SnP500priceReaction_0D_to_1D[key] = (snp500_1D - snp500_0D) / snp500_0D
            
            stock_start = stocks_startDate[key]
            stock_end = stocks_endDate[key]
            
            snp500_start = snp500_all_startDate[key]
            snp500_end = snp500_all_endDate[key]
            
#            snp500_6DMA_minus20D = (snp500_all_minus20D[key] + snp500_all_minus19D[key] + snp500_all_minus18D[key] + snp500_all_minus17D[key] + snp500_all_minus16D[key]) / 6
#            SnP500PriceChange20Day[key] = (snp500_start - snp500_6DMA_minus20D) / snp500_6DMA_minus20D
            
#            if startDate != "1D":
#            priceReactionSoFar[key] = (stock_start - stock_minus1D) / stock_minus1D
            priceReactionSoFar[key] = (stock_start - stock_minus1D) / stock_minus1D - (snp500_start - snp500_minus1D) / snp500_minus1D
            
            SnP500priceReactionSoFar[key] = (snp500_start - snp500_minus1D) / snp500_minus1D
            
 #           priceReactionForward[key] = (stock_end - stock_start) / stock_start
            priceReactionForward[key] = (stock_end - stock_start) / stock_start - (snp500_end - snp500_start) / snp500_start

            ######### DEBUG ##########
#            print("key = %s, stock_start = %f, stock_end = %f, stock_minus1D = %f, snp_start = %f, snp_end = %f, snp_minus1D = %f" % (key, stock_start, stock_end, stock_minus1D, snp500_start, snp500_end, snp500_minus1D))
#            temp = "stock_start," + str(stock_start) + "," + "stock_end," + str(stock_end) + "," + "snp_start," + str(snp500_start) + "," + "snp_end," + str(snp500_end)
#            info[key] = temp
            ######### DEBUG ##########
      
        
    if doCAR == False:
        if startDate == '-1D':
            a=1 # do nothing
        else:
            priceReactions["priceReactionSoFar"] = priceReactionSoFar
            priceReactions["SnP500priceReactionSoFar"] = SnP500priceReactionSoFar
    
        '''
        Including the price movement from -1D to 1D
        '''
        if includeMinus1Dto1D == True:
            priceReactions["priceReaction_minus1D_to_1D"] = priceReaction_minus1D_to_1D

    else:
        
        startIdx = -1
        for i in range(len(Stock_Price_DateScenarios2)):
            if Stock_Price_DateScenarios2[i] == startDate:
                startIdx = i
                
        endIdx = -1
        for i in range(len(Stock_Price_DateScenarios2)):
            if Stock_Price_DateScenarios2[i] == endDate:
                endIdx = i
                
        for dataPointKey in commonKeys:
            CAR = 0.0
            CAR_available = True
            for i in range(startIdx, endIdx):
                dateKey_t0 = Stock_Price_DateScenarios2[i]
                dateKey_t1 = Stock_Price_DateScenarios2[i+1]
                
                # This returns the stock price for all data points on a day, such as '1D'
                stockPrices_t0 = allStockPrices_TupleFormat[dateKey_t0]
                stockPrices_t1 = allStockPrices_TupleFormat[dateKey_t1]
                                
                # This returns the SnP500 price for all data points on a day, such as '1D'
                snpPrices_t0 = SnP500Prices[dateKey_t0]
                snpPrices_t1 = SnP500Prices[dateKey_t1]
                
                # Extract stock price and SnP500 price for the current data point
                stockPrice_t0 = 0
                stockPrice_t1 = 0
                snpPrice_t0 = 0
                snpPrice_t1 = 0
                if dataPointKey in stockPrices_t0 and \
                dataPointKey in stockPrices_t1 and \
                dataPointKey in snpPrices_t0 and \
                dataPointKey in snpPrices_t1:
                    stockPrice_t0 = stockPrices_t0[dataPointKey]
                    stockPrice_t1 = stockPrices_t1[dataPointKey]
                
                    snpPrice_t0 = snpPrices_t0[dataPointKey]
                    snpPrice_t1 = snpPrices_t1[dataPointKey]
                else:
                    # As soon as one alpha can't be calculated, no CAR for this data point
                    CAR_available = False
                    break
                
                one_day_alpha = (stockPrice_t1 - stockPrice_t0) / stockPrice_t0 - (snpPrice_t1 - snpPrice_t0) / snpPrice_t0
                CAR = CAR + one_day_alpha 
            
            if CAR_available == True:
                priceReactionForward[dataPointKey] = CAR
        
#        '''
#        4. Calculate the daily price %change of both company stocks and S&P500 and then the alpha return
#        '''
#        alphas_allScenarios_value = []
#        alphas_allScenarios_keys = []
#        for i in range(1, len(Stock_Price_DateScenarios3)):
#            scenarioName = Stock_Price_DateScenarios3[i]
#            lastScenarioName = Stock_Price_DateScenarios3[i-1]
#            
#            lastClosing_stock = allStockPrices_TupleFormat[lastScenarioName]
#            lastClosing_snp = SnP500Prices[lastScenarioName]
#            
#            todayClosing_stock = allStockPrices_TupleFormat[scenarioName]
#            todayClosing_snp = SnP500Prices[scenarioName]
#            
#            alphas = {}
#            for key in commonKeys:
#                if key in lastClosing_stock and key in lastClosing_snp and key in todayClosing_stock and key in todayClosing_snp:
#                    lastPrice_stock = lastClosing_stock[key]
#                    lastPrice_snp = lastClosing_snp[key]
#                    todayPrice_stock = todayClosing_stock[key]
#                    todayPrice_snp = todayClosing_snp[key]
#                    
#                    alpha = (todayPrice_stock - lastPrice_stock) / lastPrice_stock - (todayPrice_snp - lastPrice_snp) / lastPrice_snp
#                    alphas[key] = alpha
#            
#            alphas_allScenarios_value.append(alphas)
#            alphas_allScenarios_keys.append(scenarioName)
#            alphas_allScenarios[scenarioName] = alphas
#        
#        '''
#        5. Calculate Cumulative Abnormal Returns (CAR) for each day range from the day before release
#           CAR is only calculated up to the previous business day
#           For example, if the date scenario is 2D, CAR is calculated up to 1D
#        '''
#        for i in range(1, len(alphas_allScenarios_value)):
#            CARS = {}
#            scenarioName = Stock_Price_DateScenarios3[i+1]
#            
#            for key in commonKeys:
#                CAR = 0.0
#                goodKey = True
#                for j in range(i):
#                    alphas = alphas_allScenarios_value[j]
#                    keys = list(alphas.keys())
#                    if key in keys:
#                        alpha = alphas[key]
#                        CAR = CAR + alpha
#                    else:
#                        goodKey = False
#                        break
#                
#                if goodKey == True:
#                    CARS[key] = CAR
#                
#            CAR_allScenarios[scenarioName] = CARS

    print("--- GetALLStockPriceData: %f minutes ---/n" % float((time.time() - start_time) / 60.0))



#priceReactions = {} 
#priceReactionForward = {}
#scenario = {"start" : "-1D", "end" : "30D"}
#GetALLStockPriceData(scenario, priceReactions, priceReactionForward)





def GetALLVolumeData(allVolumeData_TupleFormat, allVolumeData_DictFormat, priceChanges_From1DPriorRelease_TupleForm, priceChanges_From1DPriorRelease_DictForm, priceChanges_From2Dto1DPrior_TupleForm, priceChanges_From2Dto1DPrior_DictForm):

    print("--- Start GetALLVolumeData ---")
    
    start_time = time.time()
    volume_DateScenarios = ['Pre release - 2D', 'Pre release - 1D', '1D', '2D', '3D', '4D', '5D']                
    volume_Workbook = r'Historical Volume_Static.xlsm'
    DataDir = ut.GetVolumeDataDir()

    # Parse the volume data workbook
    fullWorkbookDir = DataDir + '\\' + volume_Workbook
    volumeData = pd.ExcelFile(fullWorkbookDir)
        
    # Go through each date scenario and extract volume data
    for scenarioName in volume_DateScenarios:
        
        volumeData_currentScenario = {}
        volumeData_AllQuarters = {}
       
        # for bookData in allBookData:
        sheetData = volumeData.parse(scenarioName)
        sheetData.columns = sheetData.iloc[0]           # Use the company names as column names
        
        quarterNames = sheetData.index.tolist()
        quarterNames.pop(0)
        companyNames = sheetData.columns.tolist()
                    
        for companyName in companyNames:
            quarterData = {}
            for quarterName in quarterNames:
                value = sheetData.loc[quarterName][companyName]
                quarterName = ut.transformQuarterName(quarterName)
                
                if isinstance(value, numbers.Number):
                    if value != "#N/A N/A" and value != "nan" and not pd.isna(value) and not pd.isnull(value) :
                        volumeData_currentScenario[(companyName, quarterName)] = value
                        quarterData[quarterName] = value
            
            volumeData_AllQuarters[companyName] = quarterData
        
        allVolumeData_TupleFormat[scenarioName] = volumeData_currentScenario
        allVolumeData_DictFormat[scenarioName] = volumeData_AllQuarters
    
    # Compute % volume change after earnings release from 1D before release
    preReleasePrices_1D = allVolumeData_DictFormat['Pre release - 1D']
    for key, data in allVolumeData_DictFormat.items():
        if key != 'Pre release - 1D' and key != 'Pre release - 2D':
            priceChanges_DictForm = {}
            priceChanges_TupleForm = {}
            CalculateStockPriceChangeRatios(preReleasePrices_1D, data, priceChanges_DictForm, priceChanges_TupleForm)        
            priceChanges_From1DPriorRelease_DictForm[key] = priceChanges_DictForm
            priceChanges_From1DPriorRelease_TupleForm[key] = priceChanges_TupleForm
    
    # Compute % volume from 2D prior release to 1D prior release
    preReleasePrices_2D = allVolumeData_DictFormat['Pre release - 2D']
    CalculateStockPriceChangeRatios(preReleasePrices_2D, preReleasePrices_1D, priceChanges_From2Dto1DPrior_DictForm, priceChanges_From2Dto1DPrior_TupleForm)        
        
    print("--- GetALLVolumeData: %f minutes ---/n" % float((time.time() - start_time) / 60.0))
                
        
    
    

    
    
    
def GetALLTechnicalData(allTechnicalData_TupleFormat, scenario):

    print("--- Start GetALLTechnicalData ---")
    
    start = scenario['start']
        
#    if start == '-1D':
#        start = '0D'
    
    
    '''
    This is a big HACK.
    I only have one set of technical indicators
    '''
    start = '0D'
    
    
    technical_Workbook = r'Technical_Indicators_' + start + '.xlsm'
    
    start_time = time.time()
    technical_DateScenarios_1 = ['RSI9-static', 'RSI30-static']                
    technical_DateScenarios_2 = ['5DMA-static', '50DMA-static', '200DMA-static']                
    
    DataDir = ut.GetTechnicalDataDir()

    # Parse the volume data workbook
    fullWorkbookDir = DataDir + '\\' + technical_Workbook
    technicalData = pd.ExcelFile(fullWorkbookDir)
        
    # Go through each RSI technical data and extract it
    for scenarioName in technical_DateScenarios_1:
        
        technicalData_currentScenario = {}
        #technicalData_AllQuarters = {}
       
        # for bookData in allBookData:
        sheetData = technicalData.parse(scenarioName)
        sheetData.columns = sheetData.iloc[0]           # Use the company names as column names
        
        quarterNames = sheetData.index.tolist()
        quarterNames.pop(0)
        companyNames = sheetData.columns.tolist()
                    
        for companyName in companyNames:
            quarterData = {}
            for quarterName in quarterNames:
                value = sheetData.loc[quarterName][companyName]
                quarterName = ut.transformQuarterName(quarterName)
                
                if isinstance(value, numbers.Number):
                    if value != "#N/A N/A" and value != "nan" and not pd.isna(value) and not pd.isnull(value) :
                        technicalData_currentScenario[(companyName, quarterName)] = value
                        quarterData[quarterName] = value
            
            #technicalData_AllQuarters[companyName] = quarterData
        
        if scenarioName == 'RSI30-static':
            scenarioName = 'RSI-30D'
        elif scenarioName == 'RSI9-static':
            scenarioName = 'RSI-9D'
        allTechnicalData_TupleFormat[scenarioName] = technicalData_currentScenario
        
       
    # Prepare for the three DMA data
    DMA_5 = {}
    DMA_50 = {}
    DMA_200 = {}
    keys = set()
    for scenarioName in technical_DateScenarios_2:
        
        technicalData_currentScenario = {}
       
        # for bookData in allBookData:
        sheetData = technicalData.parse(scenarioName)
        sheetData.columns = sheetData.iloc[0]           # Use the company names as column names
        
        quarterNames = sheetData.index.tolist()
        quarterNames.pop(0)
        companyNames = sheetData.columns.tolist()
                    
        for companyName in companyNames:
            for quarterName in quarterNames:
                value = sheetData.loc[quarterName][companyName]
                quarterName = ut.transformQuarterName(quarterName)
                
                if isinstance(value, numbers.Number):
                    if value != "#N/A N/A" and value != "nan" and not pd.isna(value) and not pd.isnull(value) :
                        technicalData_currentScenario[(companyName, quarterName)] = value
                        keys.add((companyName, quarterName))
            
        if scenarioName == '5DMA-static':
            DMA_5 = technicalData_currentScenario
        elif scenarioName == '50DMA-static':
            DMA_50 = technicalData_currentScenario
        elif scenarioName == '200DMA-static':
            DMA_200 = technicalData_currentScenario
    
    # Calculate the relativeness between 5D RMA, 50D RMA and 200D RMA    
    DMA_5_50 = {}
    DMA_5_200 = {}
    DMA_50_200 = {}
    for key in keys:
        if key in DMA_5 and key in DMA_50:
            DMA_5_50[key] = DMA_5[key] / DMA_50[key]
            
        if key in DMA_5 and key in DMA_200:
            DMA_5_200[key] = DMA_5[key] / DMA_200[key]
            
        if key in DMA_50 and key in DMA_200:
            DMA_50_200[key] = DMA_50[key] / DMA_200[key]
            
    allTechnicalData_TupleFormat["DMA 5D/50D"] = DMA_5_50
    allTechnicalData_TupleFormat["DMA 5D/200D"] = DMA_5_200
    allTechnicalData_TupleFormat["DMA 50D/200D"] = DMA_50_200
    
    print("--- GetALLTechnicalData: %f minutes ---/n" % float((time.time() - start_time) / 60.0))

        
#allTechnicalData_TupleFormat = {}
#scenario = {"start" : "-1D", "end" : "10D"}
#GetALLTechnicalData(allTechnicalData_TupleFormat, scenario)   


def GetALLMacroData(allMacroData_TupleFormat, allMacroData_DictFormat):

    print("--- Start GetALLMacroData ---")
    
    start_time = time.time()
    macro_Workbook = r'Macro_Static.xlsm'
    DataDir = ut.GetMacroDataDir()

    # Parse the volume data workbook
    fullWorkbookDir = DataDir + '\\' + macro_Workbook
    macroData = pd.ExcelFile(fullWorkbookDir)
        
    # Go through each date scenario and extract volume data
    for scenarioName in macroData.sheet_names:
        
        macroData_currentScenario = {}
        macroData_AllQuarters = {}
       # for bookData in allBookData:
        sheetData = macroData.parse(scenarioName)
        sheetData.columns = sheetData.iloc[0]               # Use the company names as column names
        
        quarterNames = sheetData.index.tolist()
        quarterNames.pop(0)
        companyNames = sheetData.columns.tolist()
                
        for companyName in companyNames:
            quarterData = {}
            for quarterName in quarterNames:
                value = sheetData.loc[quarterName][companyName]
                quarterName = ut.transformQuarterName(quarterName)
                
                if isinstance(value, numbers.Number):
                    if value != "#VALUE!" and value != "#N/A N/A" and value != "nan" and not pd.isna(value) and not pd.isnull(value) :
                        macroData_currentScenario[(companyName, quarterName)] = value
                        quarterData[quarterName] = value
            
            macroData_AllQuarters[companyName] = quarterData
        
        allMacroData_TupleFormat[scenarioName] = macroData_currentScenario
        allMacroData_DictFormat[scenarioName] = macroData_AllQuarters
    
    print("--- GetALLMacroData: %f minutes ---/n" % float((time.time() - start_time) / 60.0))


def GetReleaseDates(releaseDateData):

    print("--- Start GetReleaseDates ---")
    
    start_time = time.time()
    
    DataDir = ut.GetReleaseDateTimeDir()

    # Parse the volume data workbook
    fullWorkbookDir = DataDir + '\\' + r"Release Date.xlsm"
    releaseDateWorkbook = pd.ExcelFile(fullWorkbookDir)
        
    worksheet = "Earnings Release Dates"
       
    sheetData = releaseDateWorkbook.parse(worksheet)
    sheetData.columns = sheetData.iloc[0]           # Use the company names as column names
    
    quarterNames = sheetData.index.tolist()
    quarterNames.pop(0)
    companyNames = sheetData.columns.tolist()
                
    for companyName in companyNames:

        for quarterName in quarterNames:
            value = sheetData.loc[quarterName][companyName]
            quarterName = ut.transformQuarterName(quarterName)
            
            if isinstance(value, datetime.datetime):
                if value != "#N/A N/A" and value != "nan" and not pd.isna(value) and not pd.isnull(value) :
                    releaseDateData[(companyName, quarterName)] = value
               
    print("--- GetReleaseDates: %f minutes ---/n" % float((time.time() - start_time) / 60.0))
    

 

def GetMarketCaps(marketCaps):

    print("--- Start GetMarketCaps---")
    
    start_time = time.time()
    
    DataDir = ut.GetMarketCapDir()

    # Parse the volume data workbook
    fullWorkbookDir = DataDir + '\\' + r"Static - Market Capitalization.xlsm"
    releaseDateWorkbook = pd.ExcelFile(fullWorkbookDir)
        
    worksheet = "ClassificationTransposed"
       
    sheetData = releaseDateWorkbook.parse(worksheet)
    sheetData.columns = sheetData.iloc[0]           # Use the company names as column names
    
    quarterNames = sheetData.index.tolist()
    quarterNames.pop(0)
    companyNames = sheetData.columns.tolist()
                
    for companyName in companyNames:

        for quarterName in quarterNames:
            value = sheetData.loc[quarterName][companyName]
            quarterName = ut.transformQuarterName(quarterName)
            
            #print("companyName=%s, quarterName=%s" % (companyName, quarterName))
            if value != "#N/A N/A" and value != "nan" and not pd.isna(value) and not pd.isnull(value) :
                marketCaps[(companyName, quarterName)] = value
               
    print("--- GetMarketCaps: %f minutes ---/n" % float((time.time() - start_time) / 60.0))


#marketCaps={}
#GetMarketCaps(marketCaps)
    
    

def GetAllSectorData(allSectorData):
    
    print("--- Start GetAllSectorData ---")
    
    sector_Workbook = r'Sector Information.xlsm'
    DataDir = ut.GetSectorDataDir()
    
    # Parse workbook and worksheet
    fullWorkbookDir = DataDir + '\\' + sector_Workbook
    sectorData = pd.ExcelFile(fullWorkbookDir).parse("Static")   
    
    sectorData_CompanyName = sectorData.Company.tolist()
    sectorData_Sector = sectorData.INDUSTRY_SECTOR.tolist()
    sectorData_Group = sectorData.INDUSTRY_GROUP.tolist()
    sectorData_Subgroup = sectorData.INDUSTRY_SUBGROUP.tolist()    
    
    allSectorData["CompanyName"] = sectorData_CompanyName
    allSectorData["Sector"] = sectorData_Sector
    allSectorData["Group"] = sectorData_Group
    allSectorData["Subgroup"] =  sectorData_Subgroup
    
    grouped_bySectorAndGroup = {}
    grouped_bySector = {}
    for idx in range(len(sectorData_CompanyName)):
        sector = sectorData_Sector[idx]
        group = sectorData_Group[idx]
        company = sectorData_CompanyName[idx]
        
        key = (sector, group)
        value = []
        if key in grouped_bySectorAndGroup:
            value = grouped_bySectorAndGroup[key]
        value.append(company)
        grouped_bySectorAndGroup[key] = value
    
        value2 = []
        if sector in grouped_bySector:
            value2 = grouped_bySector[sector]
        value2.append(company)
        grouped_bySector[sector] = value2
    
    allSectorData["GroupedBySectorAndGroup"] =  grouped_bySectorAndGroup
    allSectorData["GroupedBySector"] =  grouped_bySector    
    
         
    
    
def GetAllEarningsData(allEarningsData):

    print("--- Start GetAllEarningsData ---")
    
    start_time = time.time()
    
    INCLUDE_MISSING_DATA = False
    
    guidanceWorkbooks = ['Diluted EPS Adj.xlsm']  # 'Revenue.xlsm', 'Gross Margin.xlsm']
    DataDir = ut.GetEarningsDataDir()
    
    # Parse workbook and worksheet
    for guidanceWorkbook in guidanceWorkbooks:
        fullWorkbookDir = DataDir + '\\' + guidanceWorkbook
        book = pd.ExcelFile(fullWorkbookDir)
        earningsSurprise = book.parse("Earnings Surprise")   
        if guidanceWorkbook == 'Diluted EPS Adj.xlsm':
            EPS_Surprise_Backward_Diff = book.parse("EPS Surprise Backward Diff")   
            EPS_Surprise_Backward_Ave_Diff = book.parse("EPS Surprise Backward Ave Diff")          
#        guidanceSurprise = pd.ExcelFile(fullWorkbookDir).parse("Guidance Surprise")   
        
        companies = earningsSurprise.columns.tolist()
        companies = companies[1:]
        
        quarterList = ["FQ4 2018","FQ3 2018","FQ2 2018",	"FQ1 2018",	"FQ4 2017",	"FQ3 2017",	"FQ2 2017",	"FQ1 2017",	"FQ4 2016",	"FQ3 2016",	"FQ2 2016",	"FQ1 2016",	"FQ4 2015",	"FQ3 2015",	"FQ2 2015",	"FQ1 2015",	"FQ4 2014",	"FQ3 2014",	"FQ2 2014",	"FQ1 2014",	"FQ4 2013",	"FQ3 2013",	"FQ2 2013",	"FQ1 2013",	"FQ4 2012",	"FQ3 2012",	"FQ2 2012",	"FQ1 2012",	"FQ4 2011",	"FQ3 2011",	"FQ2 2011",	"FQ1 2011",	"FQ4 2010",	"FQ3 2010",	"FQ2 2010",	"FQ1 2010",	"FQ4 2009",	"FQ3 2009",	"FQ2 2009",	"FQ1 2009",	"FQ4 2008",	"FQ3 2008",	"FQ2 2008",	"FQ1 2008",	"FQ4 2007",	"FQ3 2007",	"FQ2 2007",	"FQ1 2007",	"FQ4 2006",	"FQ3 2006",	"FQ2 2006",	"FQ1 2006",	"FQ4 2005",	"FQ3 2005",	"FQ2 2005",	"FQ1 2005",	"FQ4 2004",	"FQ3 2004",	"FQ2 2004",	"FQ1 2004",	"FQ4 2003",	"FQ3 2003",	"FQ2 2003",	"FQ1 2003",	"FQ4 2002",	"FQ3 2002",	"FQ2 2002",	"FQ1 2002",	"FQ4 2001",	"FQ3 2001",	"FQ2 2001",	"FQ1 2001",	"FQ4 2000",	"FQ3 2000",	"FQ2 2000",	"FQ1 2000",	"FQ4 1999",	"FQ3 1999",	"FQ2 1999",	"FQ1 1999",	"FQ4 1998",	"FQ3 1998",	"FQ2 1998",	"FQ1 1998",	"FQ4 1997",	"FQ3 1997",	"FQ2 1997",	"FQ1 1997",	"FQ4 1996",	"FQ3 1996",	"FQ2 1996",	"FQ1 1996"]
        
        earningsSurpriseData = {}
        EPS_Surprise_Backward_Diff_Data = {}
        EPS_Surprise_Backward_Ave_Diff_Data = {}
#        guidanceSurpriseData = {}
#        earningsAndGuidanceSurprise = {}
        
        earningsSurprise_NoData = []
#        guidanceSurprise_NoData = []
#        earningsAndGuidanceSurprise_NoData = []
        
        for i in range(len(quarterList)):
            quarter = quarterList[i]
            quarter = ut.transformQuarterName(quarter)            
            
            for j in range(len(companies)):
                company = companies[j]
                key = (company, quarter)

                #print("quarter=%s, company=%s" % (quarter, company))
                
#                hasEarnings = False
#                hasGuidance = False
                
                ################################################
                # Earnings surprise
                value = earningsSurprise.iloc[i][j+1]
                if value != "" and value != 0 and value != '0' and value != '—' and value != "#N/A N/A" and value != "nan" and not pd.isna(value) and not pd.isnull(value):            
                    earningsSurpriseData[key] = value
#                    hasEarnings = True
                else:
                    if INCLUDE_MISSING_DATA == True:
                        value = 'nan'
                        earningsSurpriseData[key] = value
#                        hasEarnings = True
                    earningsSurprise_NoData.append(key)
                    
                ################################################
                # EPS surprise backward diff and backward average diff
                if guidanceWorkbook == 'Diluted EPS Adj.xlsm':
                    # EPS surprise backward diff
                    value = EPS_Surprise_Backward_Diff.iloc[i][j+1]
                    if value != "" and value != 0 and value != '0' and value != '—' and value != "#N/A N/A" and value != "nan" and not pd.isna(value) and not pd.isnull(value):            
                        EPS_Surprise_Backward_Diff_Data[key] = value                    
                    else:
                        if INCLUDE_MISSING_DATA == True:
                            value = 'nan'
                            EPS_Surprise_Backward_Diff_Data[key] = value                        
                            
                    # EPS surprise backward average diff
                    value = EPS_Surprise_Backward_Ave_Diff.iloc[i][j+1]
                    if value != "" and value != 0 and value != '0' and value != '—' and value != "#N/A N/A" and value != "nan" and not pd.isna(value) and not pd.isnull(value):            
                        EPS_Surprise_Backward_Ave_Diff_Data[key] = value                    
                    else:
                        if INCLUDE_MISSING_DATA == True:
                            value = 'nan'
                            EPS_Surprise_Backward_Ave_Diff_Data[key] = value                        
                
#                ###############################################                       
                 #Guidance surprise
#                value2 = guidanceSurprise.iloc[i][j+1]        
#                if value2 != "" and value2 != 0 and value2 != '0' and value2 != '—' and value2 != "#N/A N/A" and value2 != "nan" and not pd.isna(value2) and not pd.isnull(value2):            
#                    guidanceSurpriseData[key] = value2
#                    hasGuidance = True
#                else:
#                    if INCLUDE_MISSING_DATA == True:
#                        value = 'nan'
#                        guidanceSurpriseData[key] = value
#                        hasGuidance = True
#                    guidanceSurprise_NoData.append(key)  
#                    
#                # Earnings surprise and Guidance surprise
#                if hasGuidance == True and hasEarnings == True:
#                    earningsAndGuidanceSurprise[key] = [value, value2]
#                else:
#                    earningsAndGuidanceSurprise_NoData.append(key)
    
        prefix = ""
        if guidanceWorkbook == 'Revenue.xlsm':
            prefix = 'revenue'
        elif guidanceWorkbook == 'Diluted EPS Adj.xlsm':
            prefix = 'EPS'
        elif guidanceWorkbook == 'Gross Margin.xlsm':
            prefix = 'Gross Margin'
            
        allEarningsData[prefix + " EarningsSurprise"] = earningsSurpriseData
#        allEarningsData[prefix + " GuidanceSurprise"] = guidanceSurpriseData    
        allEarningsData[prefix + " EarningsSurprise_NoData"] = earningsSurprise_NoData
#        allEarningsData[prefix + " GuidanceSurprise_NoData"] = guidanceSurprise_NoData
#        allEarningsData[prefix + " EarningsAndGuidanceSurprise"] = earningsAndGuidanceSurprise
#        allEarningsData[prefix + " EarningsAndGuidanceSurprise_NoData"] = earningsAndGuidanceSurprise_NoData
        
        if guidanceWorkbook == 'Diluted EPS Adj.xlsm':
            allEarningsData[prefix + " Earnings_Surprise_Backward_Diff"] = EPS_Surprise_Backward_Diff_Data
            allEarningsData[prefix + " Earnings_Surprise_Backward_Ave_Diff"] = EPS_Surprise_Backward_Ave_Diff_Data
            
    print("--- GetAllEarningsData: %f minutes ---/n" % float((time.time() - start_time) / 60.0))


def StandardizeDataPoints(allDataPoints, allDataPointNames):
    start_time = time.time()
    # Randomly select an item from 'allDataPoints'
    attributeNames = allDataPoints[list(allDataPoints.keys())[0]].keys()
                
    allStandardizedDataPoints_TEMP = []
    for key, value in allDataPoints.items():
        allDataPointNames.append(key)
        dataPoint = []
        for attributeName in attributeNames:
            dataPoint.append(value[attributeName])
        
        allStandardizedDataPoints_TEMP.append(dataPoint)
    
    scaler = RobustScaler()
    scaler.fit(allStandardizedDataPoints_TEMP)
    allStandardizedDataPoints = scaler.transform(allStandardizedDataPoints_TEMP).tolist()
    
    print("--- StandardizeDataPoints: %f minutes ---/n" % float((time.time() - start_time) / 60.0))
    return allStandardizedDataPoints 
    



def GetTrainingData_OneScenario(allReportDataPoints, 
                                allTechnicalData, 
                                allShortInterestRatios,
                                allSectorData, 
                                allEarningsData,
                                releaseDateData,
                                marketCaps,
                                priceReactions, 
                                priceReactionForward,
                                use_Data_From_Year,
                                exclude2008Crsis):
    
    print("--- Start GetTrainingData_OneScenario ---")
    start_time = time.time()
    dataPointNames = []
    featureNames = []
    
    '''
    dataPointNames is a collection of (companyName, quarterName)
    In the end 'dataPointNames' will be the common subset of (companyName, quarterName) that exist in all the features
    '''
    
    # Get keys for 'priceReactionForward'. 
    # 'priceReactionForward' are the outputs of the learning models.
    stockPriceChanges_keys = priceReactionForward.keys()
    dataPointNames = stockPriceChanges_keys
    
    ###############################################
    # 1. Find data point names that exist in all types of data
    
    # reports data
    reportData_keys = allReportDataPoints.keys()
    dataPointNames = list(set(dataPointNames).intersection(reportData_keys))
    featureNames = list(list(allReportDataPoints.values())[0].keys())
    
    # technical
    # For now, all technical data are given on the day the earnings are released
    technical_metric_keys = []
    technical_metric_keys = allTechnicalData.keys()
    for metricName in technical_metric_keys:
        technical_metric = allTechnicalData[metricName]
        keys = technical_metric.keys()
        dataPointNames = list(set(dataPointNames).intersection(keys))   
        featureNames.append(metricName)
        
    # Short Interest Ratios
    shortRatioKeys = allShortInterestRatios.keys()
    dataPointNames = list(set(dataPointNames).intersection(shortRatioKeys))   
    featureNames.append("Short Interest Ratios")
    
    # Earnings Surprise data
    earnings_metric_keys = []
    earnings_metric_keys = allEarningsData.keys()
    for metricName in earnings_metric_keys:
        if metricName == 'EPS EarningsSurprise' or metricName == 'EPS Earnings_Surprise_Backward_Diff' or metricName == 'EPS Earnings_Surprise_Backward_Ave_Diff':
            metric = allEarningsData[metricName]
            keys = metric.keys()
            dataPointNames = list(set(dataPointNames).intersection(keys))   
            featureNames.append(metricName)
    
    # price reactions
    priceReaction_metric_keys = []
    priceReaction_metric_keys = priceReactions.keys()
    for metricName in priceReaction_metric_keys:
        priceReaction_metric = priceReactions[metricName]
        if priceReaction_metric != {}:
            keys = priceReaction_metric.keys()
            dataPointNames = list(set(dataPointNames).intersection(keys))   
            featureNames.append(metricName)
    
    # Prediction outputs    
    featureNames.append("Outputs_Regression")
    featureNames.append("Outputs_Classification")
    
    # Earning release dates
    releaseDateKeys = releaseDateData.keys()
    dataPointNames = list(set(dataPointNames).intersection(releaseDateKeys))   
    featureNames.append("Release Date")
    
    # Market capitalization
    marketCapKeys = marketCaps.keys()
    dataPointNames = list(set(dataPointNames).intersection(marketCapKeys))   
    featureNames.append("Market Cap")
    
    
    ###############################################
    ## 2. Get the input and output points ##
    
    # The data points are the last two quarters of companies that ceased to trade
    dataPointsToExclude = [('BCR', '2017 FQ3'),	('BCR', '2017 FQ2'),	('AAMRQ', '2013 FQ2'),	('AAMRQ', '2013 FQ3'),	('ARB', '2013 FQ1'),	('ARB', '2013 FQ2'),	('BDK', '2009 FQ4'),	('BDK', '2009 FQ3'),	('BEAM', '2013 FQ4'),	('BEAM', '2013 FQ3'),	('BHI', '2017 FQ1'),	('BHI', '2016 FQ4'),	('BLS', '2006 FQ3'),	('BLS', '2006 FQ2'),	('BOL', '2005 FQ3'),	('BOL', '2005 FQ2'),	('CBE', '2012 FQ3'),	('CBE', '2012 FQ2'),	('CCTYQ', '2009 FQ1'),	('CCTYQ', '2008 FQ4'),	('CEG', '2011 FQ4'),	('CEG', '2011 FQ3'),	('CMCSK', '2015 FQ3'),	('CMCSK', '2015 FQ2'),	('CNW', '2015 FQ2'),	('CNW', '2015 FQ1'),	('COMS', '2010 FQ2'),	('COMS', '2010 FQ1'),	('CSC', '2017 FQ3'),	('CSC', '2017 FQ2'),	('DALRQ', '2006 FQ1'),	('DALRQ', '2005 FQ2'),	('DCNAQ', '2005 FQ3'),	('DCNAQ', '2005 FQ2'),	('DD', '2017 FQ2'),	('DD', '2017 FQ1'),	('DELL', '2014 FQ2'),	('DELL', '2014 FQ1'),	('DJ', '2007 FQ3'),	('DJ', '2007 FQ2'),	('DOW', '2017 FQ2'),	('DOW', '2017 FQ1'),	('EKDKQ', '2013 FQ2'),	('EKDKQ', '2011 FQ3'),	('EKDKQ', '2011 FQ2'),	('EMC', '2016 FQ2'),	('EMC', '2016 FQ1'),	('ETS', '2005 FQ3'),	('ETS', '2005 FQ2'),	('FWLT', '2014 FQ3'),	('FWLT', '2014 FQ2'),	('GAPTQ', '2011 FQ2'),	('GAPTQ', '2011 FQ1'),	('GDW', '2006 FQ2'),	('GDW', '2006 FQ1'),	('GP', '2005 FQ3'),	('GP', '2005 FQ2'),	('GR', '2012 FQ1'),	('GR', '2011 FQ4'),	('HAR', '2017 FQ2'),	('HAR', '2017 FQ1'),	('HET', '2007 FQ3'),	('HET', '2007 FQ2'),	('HNZ', '2013 FQ3'),	('HNZ', '2013 FQ1'),	('HOT', '2016 FQ2'),	('HOT', '2016 FQ1'),	('HSH', '2014 FQ4'),	('HSH', '2014 FQ3'),	('IKN', '2008 FQ3'),	('IKN', '2008 FQ2'),	('JAVA', '2010 FQ1'),	('JAVA', '2009 FQ4'),	('JH', '2006 FQ4'),	('JH', '2006 FQ3'),	('KATE', '2017 FQ1'),	('KATE', '2016 FQ4'),	('KRI', '2006 FQ1'),	('KRI', '2005 FQ4'),	('LLTC', '2017 FQ2'),	('LLTC', '2017 FQ1'),	('MEE', '2010 FQ4'),	('MEE', '2010 FQ3'),	('MEL', '2006 FQ4'),	('MEL', '2006 FQ3'),	('MER', '2008 FQ3'),	('MER', '2008 FQ2'),	('MTLQQ', '2008 FQ4'),	('MTLQQ', '2008 FQ3'),	('MZIAQ', '2008 FQ2'),	('MZIAQ', '2008 FQ1'),	('NCC', '2008 FQ3'),	('NCC', '2008 FQ2'),	('NOVL', '2010 FQ4'),	('NOVL', '2010 FQ3'),	('OMX', '2013 FQ2'),	('OMX', '2013 FQ1'),	('PAS', '2009 FQ3'),	('PAS', '2009 FQ2'),	('PD', '2006 FQ3'),	('PD', '2006 FQ2'),	('PLL', '2015 FQ2'),	('PLL', '2015 FQ1'),	('RAI', '2017 FQ1'),	('RAI', '2016 FQ4'),	('RHDCQ', '2009 FQ2'),	('RHDCQ', '2009 FQ1'),	('RML', '2005 FQ4'),	('RML', '2005 FQ3'),	('RSHCQ', '2015 FQ2'),	('RSHCQ', '2015 FQ1'),	('SGP', '2009 FQ2'),	('SGP', '2009 FQ1'),	('SIAL', '2015 FQ2'),	('SIAL', '2015 FQ1'),	('SNI', '2017 FQ3'),	('SNI', '2017 FQ2'),	('SRR', '2007 FQ1'),	('SRR', '2006 FQ4'),	('STJ', '2016 FQ2'),	('STJ', '2016 FQ1'),	('TEK', '2007 FQ4'),	('TEK', '2007 FQ3'),	('TIN', '2011 FQ2'),	('TIN', '2011 FQ1'),	('TLAB', '2013 FQ2'),	('TLAB', '2013 FQ1'),	('TNB', '2011 FQ4'),	('TNB', '2011 FQ3'),	('TRB', '2007 FQ3'),	('TRB', '2007 FQ2'),	('TWX', '2018 FQ1'),	('TWX', '2017 FQ4'),	('TXU', '2007 FQ2'),	('TXU', '2006 FQ3'),	('UCL', '2005 FQ1'),	('UCL', '2004 FQ4'),	('UIS', '2018 FQ1'),	('UIS', '2017 FQ4'),	('VFC', '2017 FQ4'),	('VFC', '2017 FQ3'),	('WFM', '2017 FQ3'),	('WFM', '2017 FQ2'),	('WYE', '2009 FQ1'),	('WYE', '2008 FQ4')]
    
    # These data points are selected because their data were released within 30 min of the close of the trading day. Previously I included them and assumed the release
    # happened AFTER the market close but I think this assumption is flawed and hence I've decided to not use these data points at all
    dataPoitnsToExclude2 = [('SYK', '2012 FQ2'),	('SYK', '2012 FQ3'),	('SYK', '2012 FQ1'),	('SYK', '2011 FQ4'),	('SYK', '2011 FQ3'),	('SYK', '2011 FQ2'),	('STL', '2012 FQ4'),	('SRPT', '2011 FQ1'),	('NSC', '2010 FQ4'),	('PANW', '2016 FQ2'),	('ORCL', '2011 FQ2'),	('ORCL', '2011 FQ1'),	('NTAP', '2011 FQ2'),	('MSFT', '2011 FQ2'),	('MASI', '2010 FQ4'),	('LVS', '2010 FQ4'),	('JNPR', '2012 FQ1'),	('HPQ', '2014 FQ2'),	('HP', '2004 FQ3'),	('HAL', '2007 FQ3'),	('FCNCA', '2017 FQ3'),	('EXR', '2012 FQ1'),	('EXR', '2011 FQ4'),	('ESS', '2015 FQ2'),	('ANAT', '2017 FQ4'),	('ANAT', '2017 FQ1'),	('AMAT', '2013 FQ2'),	('AIV', '2014 FQ2'),	('HRB', '2011 FQ2'),	('CSL', '2016 FQ3'),	('CXP', '2010 FQ2'),	('COLM', '2013 FQ3'),	('DIS', '2010 FQ4'),	('ANAT', '2016 FQ3'),	('Y', '2012 FQ2'),	('Y', '2011 FQ1'),	('ANAT', '2011 FQ2'),	('ANAT', '2010 FQ1'),	('AKAM', '2007 FQ1'),	('AFL', '2014 FQ1'),	('TWOU', '2018 FQ1'),	('SNI', '2017 FQ2'),	('HNZ', '2014 FQ2'),	('VNO', '2011 FQ2'),	('TWTR', '2015 FQ1'),	('UNM', '2011 FQ4'),	('URI', '2013 FQ3'),	('ULTI', '2011 FQ4'),	('RIG', '2010 FQ4')]
    
    # These are quarters during the two financial crisis
    quartersToExclude = []
    if exclude2008Crsis == True:
        quartersToExclude = ['2007 FQ4', '2008 FQ1', '2008 FQ2', '2008 FQ3', '2008 FQ4', '2009 FQ1', '2009 FQ2', '2009 FQ3', '2009 FQ4']
    
    traingPointsByCompany = {}
    for dataPointKey in dataPointNames:
        
        companyName = dataPointKey[0]
        quarterName = dataPointKey[1]
        year = int(quarterName[:4])
        
#        if dataPointKey not in dataPointsToExclude: 
#        if dataPointKey not in dataPointsToExclude and quarterName not in quartersToExclude and year > 2003: 
        if dataPointKey not in dataPointsToExclude and \
            dataPointKey not in dataPoitnsToExclude2 and \
            dataPointKey not in quartersToExclude and \
            year > use_Data_From_Year:
                
            '''
            1. Exclude certain data points. These excluded data points are the last two quarters of companies that ceased to trade
            2. Exclude those quarters during the two financial crisis
            3. Exclude any data points before 2003
            '''
            
            '''
            2.1 Get the input points
            '''
            
            oneDataPoint = []
            
#            print("dataPointKey", dataPointKey)
#            if dataPointKey == ('ADI', '2011 FQ1'):
#                abc = 1
                                
            # 2.1.1 Insert financial report data to InputPoints
            if dataPointKey in reportData_keys:
                oneDataPoint = oneDataPoint + list(allReportDataPoints[dataPointKey].values())
            
            # 2.1.2 Insert technical data to InputPoints
            for metricName in technical_metric_keys:
                technical_metric = allTechnicalData[metricName]
                keys = technical_metric.keys()
                if dataPointKey in keys:
                    oneDataPoint.append(technical_metric[dataPointKey])
                else:
                    oneDataPoint.append('nan')
                    print("No data of type %s is found for %s" % (metricName, dataPointKey))
            
            # 2.1.3 Insert short interest ratios
            keys = allShortInterestRatios.keys()
            if dataPointKey in keys:
                oneDataPoint.append(allShortInterestRatios[dataPointKey])
            else:
                oneDataPoint.append('nan')
                print("No data of type %s is found for %s" % (metricName, dataPointKey))
            
            # 2.1.4 Insert earnings data to InputPoints
            for metricName in earnings_metric_keys:
                if metricName == 'EPS EarningsSurprise' or metricName == 'EPS Earnings_Surprise_Backward_Diff' or metricName == 'EPS Earnings_Surprise_Backward_Ave_Diff':
                    earnings_metric = allEarningsData[metricName]
                    keys = earnings_metric.keys()
                    if dataPointKey in keys:
                        oneDataPoint.append(earnings_metric[dataPointKey])
                    else:
                        oneDataPoint.append('nan')
                        print("No data of type %s is found for %s" % (metricName, dataPointKey))
            
            # 2.1.5 Insert price reaction data to InputPoints
            for metricName in priceReaction_metric_keys:
                priceReaction_metric = priceReactions[metricName]
                if priceReaction_metric != {}:
                    keys = priceReaction_metric.keys()
                    if dataPointKey in keys:
                        oneDataPoint.append(priceReaction_metric[dataPointKey])
                    else:
                        oneDataPoint.append('nan')
                        print("No data of type %s is found for %s" % (metricName, dataPointKey))
                    
            # 2.1.6 Insert output data points
        
            if dataPointKey in stockPriceChanges_keys:
                y = priceReactionForward[dataPointKey]
                oneDataPoint.append(y)
    
                binaryVal = bool()
                if y < 0:
                    binaryVal = 0
                elif y >= 0:
                    binaryVal = 1
                oneDataPoint.append(binaryVal)
            else:
                raise Exception('Stock price change is not available!!')
            
              
            # 2.1.7 Insert release date
            keys = releaseDateData.keys()
            if dataPointKey in keys:
                oneDataPoint.append(releaseDateData[dataPointKey])
            else:
                oneDataPoint.append('nan')
                print("No data of type %s is found for %s" % (metricName, dataPointKey))
                
            # 2.1.8 Insert market cap
            keys = marketCaps.keys()
            if dataPointKey in keys:
                oneDataPoint.append(marketCaps[dataPointKey])
            else:
                oneDataPoint.append('nan')
                print("No data of type %s is found for %s" % (metricName, dataPointKey))
                
                
                
            '''
            2.2 Store data into data frame for each company
            '''
            
            # Retrieve an existing company data frame if it's there
            currentCompanyData = pd.DataFrame()            
            if companyName in traingPointsByCompany:
                currentCompanyData = traingPointsByCompany[companyName]
            
            # Append current row
            s = pd.Series(oneDataPoint, index = featureNames)
            s.name = dataPointKey
            currentCompanyData = currentCompanyData.append(s)
            currentCompanyData = currentCompanyData[featureNames] # adding [featureNames] to the end forces the dataframe to use the columns and the particular order defined by featureNames
            
            traingPointsByCompany[companyName] = currentCompanyData
        
            
    ## Store all data without sector separations
    ret = {}
    ret["Feature Names"] = featureNames
    ret["Training Points"] = traingPointsByCompany



    ###############################################
    ## 3. Divide output data by sectors ##
    
    ## Divide output data by sectors and groups
    trainingPointsBySectorGroup = {}
    
    groupedBySector = allSectorData["GroupedBySector"]    
        
    # For each company, check which sector-group it belongs to, and assign its data correspondingly
    for groupName, companiesInGroup in groupedBySector.items():    
        
        val1 = {}
        if groupName in trainingPointsBySectorGroup:
            val1 = trainingPointsBySectorGroup[groupName]
        
        for company in companiesInGroup:
            
            if company in traingPointsByCompany:
                val1[company] = traingPointsByCompany[company]
                
        if val1 != {}:
            trainingPointsBySectorGroup[groupName] = val1
        
    ret["By SectorGroup - Training Points"] = trainingPointsBySectorGroup
    
    print("--- GetTrainingData_OneScenario: %f minutes ---/n" % float((time.time() - start_time) / 60.0))
    return ret
    

#trainingData_OneScenario_test = {}
#trainingData_OneScenario_test = GetTrainingData_OneScenario(allReportDataPoints, 
#                                                       allTechnicalData_TupleFormat, 
#                                                       allShortInterestRatios,
#                                                       allSectorData,                                                        
#                                                       allEarningsData,
#                                                       priceReactions, 
#                                                       priceReactionForward)


def CompanyLevelPreProcessing(trainingData, allReportDataPoints, allSectorData, useMinMaxScalingForInputs, scalingMethodForPrices, priceScalers):
    
    print("--- Start CompanyLevelPreProcessing ---")
    start_time = time.time()
    
    # Two goals: company level normalization and outlier removal
    
    newTrainingData = {}
    
    processColumnsUpto = len(allReportDataPoints[list(allReportDataPoints.keys())[0]]) 
    
    featureNames = trainingData["Feature Names"][:processColumnsUpto]
    
    trainingDataPoints_GroupedByCompany = trainingData["Training Points"]
    
    # Loop through each of the companies
    newTrainingData_GroupedByCompany = {}
    for companyName, companyDataOriginalFull in trainingDataPoints_GroupedByCompany.items():
        
        dataPointNames = companyDataOriginalFull.index
        priceData = companyDataOriginalFull.iloc[:, processColumnsUpto:]
        companyDataOriginal = companyDataOriginalFull.iloc[:, :processColumnsUpto]

        # 1. Winsorization        
        # Loop through each of the features of the current company
        companyData_winsorized = pd.DataFrame()
        for featureName in featureNames:
                                 
            featureValue = pd.Series(companyDataOriginal[featureName], index = dataPointNames)
            
            standard_deviations = {}
            
            step = 0.02
            
            # Go through all combinations of thresholds and find the new standard deviation for each threshold
            for i in range(6):
                left = i * step
                for j in range(6):                    
                    right = j * step
                    featureValue_winsorized_temp = scipy.stats.mstats.winsorize(featureValue, limits = (left, right))        
                    std = featureValue_winsorized_temp.std()
                    standard_deviations[std] = (i,j)
            
            standard_deviations_diff = {}
            
            if len(standard_deviations) > 1:
                
                # Sort the standard deviations and find the difference between i-1 and i
                lastKey = -100000
                for key in sorted(standard_deviations, reverse = True):
                    if lastKey != -100000:
                        std_diff = lastKey - key
                        standard_deviations_diff[std_diff] = standard_deviations[key]
                        
                    lastKey = key
    
                # Find the largest standard deviation diff            
                max_std_diff = max(k for k in standard_deviations_diff.keys() )
                
                optimal_threshold_index = standard_deviations_diff[max_std_diff]
                        
                i = optimal_threshold_index [0]
                j = optimal_threshold_index [1]
                
                featureValue_winsorized = scipy.stats.mstats.winsorize(featureValue, (i * step, j * step))
                featureValue_winsorized.index = dataPointNames
                
            else:
                
                featureValue_winsorized = featureValue.values
                #featureValue_winsorized.index = dataPointNames
                
            # Put new list of feature values into new Data Frame. This will form a column of the new DF.
            companyData_winsorized[featureName] = featureValue_winsorized
    
        companyData_winsorized.reset_index()
        companyData_winsorized.reindex(index = dataPointNames)
                
        # 2. Allow certain extra columns to be put through standardization
        additionalColumns = ['EPS EarningsSurprise', 'EPS Earnings_Surprise_Backward_Diff', 'EPS Earnings_Surprise_Backward_Ave_Diff']
        for columnName in additionalColumns:
            featureValue = pd.Series(companyDataOriginalFull[columnName], index = dataPointNames)
            companyData_winsorized[columnName] = featureValue.values
            priceData = priceData.drop(columnName, 1)   # remember to drop this column from 'priceData' to avoid double counting
            
        # 3. Standardization
        companyData_standardized = pd.DataFrame()
        if useMinMaxScalingForInputs == True:
            
            scaler = preprocessing.MinMaxScaler().fit(companyData_winsorized)
            companyData_standardized = pd.DataFrame(data = scaler.transform(companyData_winsorized),
                                                index = dataPointNames,
                                                columns = companyData_winsorized.columns)
        else:
            
            scaler = preprocessing.StandardScaler().fit(companyData_winsorized)
            companyData_standardized = pd.DataFrame(data = scaler.transform(companyData_winsorized),
                                                index = dataPointNames,
                                                columns = companyData_winsorized.columns)
        
#        # This line has no standardisation
#        companyData_standardized = companyData_winsorized #companyData_winsorized
#        companyData_standardized.index = dataPointNames




        # 4. Normalize stock returns
        if scalingMethodForPrices != "":
            
            stockReturns = pd.DataFrame()
            stockReturns = priceData['Outputs_Regression']
            toNormalise = []                                
            for i in range(len(stockReturns)):
                singleReturn = stockReturns[i]
                toNormalise.append([singleReturn])                            
                                       
            stockReturns_normalized = pd.DataFrame()
            if scalingMethodForPrices == "MinMax":
                scaler = preprocessing.MinMaxScaler().fit(toNormalise)                                
                stockReturns_normalized = scaler.transform(toNormalise)                                
                priceScalers[companyName] = scaler
            elif scalingMethodForPrices == "ZSpread":
                scaler = preprocessing.StandardScaler().fit(toNormalise)                                
                stockReturns_normalized = scaler.transform(toNormalise)                                
                priceScalers[companyName] = scaler
                
            stockReturns_normalized_df = pd.DataFrame(data = stockReturns_normalized, index = dataPointNames, columns = ['Outputs_Regression'])  
            priceData['Outputs_Regression'] = stockReturns_normalized_df   
                        
            
        # 5. Join the processed data back to the feature collection of the current company
        a = companyData_standardized
        b = priceData
        c = pd.concat([a, b ], axis = 1)
        newTrainingData_GroupedByCompany[companyName] = c

    newTrainingData["Training Points"] = newTrainingData_GroupedByCompany
    newTrainingData["Feature Names"] = trainingData["Feature Names"]
    
    ## Divide training data by sectors and groups
    trainingPointsBySectorGroup = {}
    groupedBySector = allSectorData["GroupedBySector"]    
        
    # For each company, check which sector-group it belongs to, and assign its data correspondingly
    for groupName, companiesInGroup in groupedBySector.items():    
        
        val1 = {}
        if groupName in trainingPointsBySectorGroup:
            val1 = trainingPointsBySectorGroup[groupName]
        
        for company in companiesInGroup:
            
            if company in newTrainingData_GroupedByCompany:
                val1[company] = newTrainingData_GroupedByCompany[company]
                
        if val1 != {}:
            trainingPointsBySectorGroup[groupName] = val1
            
    newTrainingData["By SectorGroup - Training Points"] = trainingPointsBySectorGroup
    
    print("--- CompanyLevelPreProcessing: %f minutes ---/n" % float((time.time() - start_time) / 60.0))
    
    return newTrainingData



def FlattenTrainingData(trainingData):
    
    print("--- Start FlattenTrainingData ---")
    start_time = time.time()
    
    trainingData_flat = {}
    trainingDataPoints_all = trainingData["Training Points"]
    trainingDataPoints_all_flat = pd.concat([com for com in trainingDataPoints_all.values()], axis = 0)#, ignore_index = True)
    trainingData_flat["all"] = trainingDataPoints_all_flat

    trainingDataPoints_sector = trainingData["By SectorGroup - Training Points"]
    for sectorName, sectorData in trainingDataPoints_sector.items():
        trainingData_flat[sectorName] = pd.concat([com for com in sectorData.values()], axis = 0)#, ignore_index = True)
        
    print("--- FlattenTrainingData: %f minutes ---/n" % float((time.time() - start_time) / 60.0))
    return trainingData_flat



def GetAllFinancialReportsDataPoints(allReportDataPoints, log1, log2):
    
    print("--- Start GetAllFinancialReportsDataPoints ---")
    
    start_time = time.time()
    
    quarterList = ["1996 FQ4",	"1997 FQ1",	"1997 FQ2",	"1997 FQ3",	"1997 FQ4",	"1998 FQ1",	"1998 FQ2",	"1998 FQ3",	"1998 FQ4",	"1999 FQ1",	"1999 FQ2",	"1999 FQ3",	"1999 FQ4",	"2000 FQ1",	"2000 FQ2",	"2000 FQ3",	"2000 FQ4",	"2001 FQ1",	"2001 FQ2",	"2001 FQ3",	"2001 FQ4",	"2002 FQ1",	"2002 FQ2",	"2002 FQ3",	"2002 FQ4",	"2003 FQ1",	"2003 FQ2",	"2003 FQ3",	"2003 FQ4",	"2004 FQ1",	"2004 FQ2",	"2004 FQ3",	"2004 FQ4",	"2005 FQ1",	"2005 FQ2",	"2005 FQ3",	"2005 FQ4",	"2006 FQ1",	"2006 FQ2",	"2006 FQ3",	"2006 FQ4",	"2007 FQ1",	"2007 FQ2",	"2007 FQ3",	"2007 FQ4",	"2008 FQ1",	"2008 FQ2",	"2008 FQ3",	"2008 FQ4",	"2009 FQ1",	"2009 FQ2",	"2009 FQ3",	"2009 FQ4",	"2010 FQ1",	"2010 FQ2",	"2010 FQ3",	"2010 FQ4",	"2011 FQ1",	"2011 FQ2",	"2011 FQ3",	"2011 FQ4",	"2012 FQ1",	"2012 FQ2",	"2012 FQ3",	"2012 FQ4",	"2013 FQ1",	"2013 FQ2",	"2013 FQ3",	"2013 FQ4",	"2014 FQ1",	"2014 FQ2",	"2014 FQ3",	"2014 FQ4",	"2015 FQ1",	"2015 FQ2",	"2015 FQ3",	"2015 FQ4",	"2016 FQ1",	"2016 FQ2",	"2016 FQ3",	"2016 FQ4",	"2017 FQ1",	"2017 FQ2",	"2017 FQ3",	"2017 FQ4",	"2018 FQ1",	"2018 FQ2",	"2018 FQ3",	"2018 FQ4"]
    
    badStocks = ['MANH', 'MAS', 'BBT', 'JH', 'CMD', 'CCK']  # These are stock codes that have been found to have bad download from Bloomberg
    
    financialReportMetrics_ByCompany = {}
    metricNames = []
    
    import os
    testFolder = ut.GetFinancialReportMetricsDir()
    files = os.listdir(testFolder)
    
    for fileName in files:
        
        # Unpack the current file
        metricName = fileName.replace(r"Static - ", "")
        metricName = metricName.replace(r".xlsm", "")
        metricNames.append(metricName)
        
        featureFile = testFolder + "\\" + fileName
        book = pd.ExcelFile(featureFile)
        sheetName = book.sheet_names[0]
        sheet = book.parse(sheetName)          
        
        # Pre-processing
        sheet = reports.WorksheetPreProcessing(sheet)
        
        # Structure of 'sheetDictionary' : Quarter name (key) -> company name (key) -> metric value of this company
        sheetDictionary = sheet.to_dict()
        
        # Structure of 'financialReportMetrics_ByCompany' : company name (key) -> metric Name (key) -> metric value dictionary with quarter name as key and metric value as value
        for quarterName, metricValues_AllCompanies in sheetDictionary.items():
            quarterName = ut.transformQuarterName(quarterName)
            
            for companyName, metricValue in metricValues_AllCompanies.items():
                
                # Do not include any bad stocks
                if companyName not in badStocks:
                    allMetricValues_OneCompany = {}
                    if companyName in financialReportMetrics_ByCompany:
                        allMetricValues_OneCompany = financialReportMetrics_ByCompany[companyName]
                        
                    singleMetricValues_OneCompany = {}
                    if metricName in allMetricValues_OneCompany:
                        singleMetricValues_OneCompany = allMetricValues_OneCompany[metricName]
                    
                    singleMetricValues_OneCompany[quarterName] = metricValue
                    allMetricValues_OneCompany[metricName] = singleMetricValues_OneCompany
                    financialReportMetrics_ByCompany[companyName] = allMetricValues_OneCompany
    
    # Calculate quarterly and yearly diffs
    quarterlyDiffs = reports.calculateQuarterlyDiffs(financialReportMetrics_ByCompany, quarterList)
    yearlyDiffs = reports.calculateYearlyDiffs(financialReportMetrics_ByCompany, quarterList)
    
    '''
    Important:
        This is potentially a contentious point.
        May need to experiment WITH or WITHOUT native ratio features!!!
    '''
    
    # Merge quarterly and yearly diff results AND native results of all ratio features
    financialReportMetrics_ByCompany_Final = {}
    for companyName, quarterlyCompanyMetrics in quarterlyDiffs.items():
        temp = deepcopy(quarterlyCompanyMetrics)
        
        # Merge yearly changes
        yearlyCompanyMetrics = yearlyDiffs[companyName]    
        temp.update(yearlyCompanyMetrics)
        
        # Merge ratio features
        nativeCompanyMetrics = financialReportMetrics_ByCompany[companyName]
        nativeCompanyMetrics_new = {}
        for key in ut.ratioFeatures:
            if key in nativeCompanyMetrics:
                nativeCompanyMetrics_new[key] = nativeCompanyMetrics[key]
        temp.update(nativeCompanyMetrics_new)
        
        financialReportMetrics_ByCompany_Final[companyName] = temp
    
    # Re-organize things by data points of (companyName, quarterName)
    for companyName, companyMetrics in financialReportMetrics_ByCompany_Final.items():
        for metricName, metricValues in companyMetrics.items():
            for quarterName, metricValue in metricValues.items():
                
                pointName = (companyName, quarterName)
                
                # PointValue is a dictionary of numerous metrics for the current (companyName, quarterName)
                pointValue = {}
                if pointName in allReportDataPoints:
                    pointValue = allReportDataPoints[pointName]
                    
                '''
                Important:
                    This is where 0 is filled in for those metrics which do not have a value
                '''
                
                if str(type(metricValue)) == "<class 'str'>":
                    pointValue[metricName] = 0  # Set cells of 'string' type to 0, effectively eliminating all NaNs and empty strings
                else:
                    pointValue[metricName] = metricValue
                    
                allReportDataPoints[pointName] = pointValue
                
    # Add in each data point's missing metrics and set the value of phantom metrics to 0
    myChosenFeatures = metricNames
    tempQ = [i + "_Q_Change" for i in myChosenFeatures]
    tempY = [i + "_Y_Change" for i in myChosenFeatures]

    myChosenFeatures = tempQ + tempY + ut.ratioFeatures  
    
    # Diagnosis - find out what is missing
    for dataPointName, dataPointValue in allReportDataPoints.items():            
        if len(dataPointValue) != len(myChosenFeatures):
            for chosenFeature in myChosenFeatures:
                if chosenFeature not in dataPointValue:
                    dataPointValue[chosenFeature] = 0
                    
                    # Logging
                    missingMetrics = []
                    if dataPointName in log1:
                        missingMetrics = log1[dataPointName]
                    missingMetrics.append(chosenFeature)
                    log1[dataPointName] = missingMetrics
                    
                    missingCount = 0
                    if dataPointName in log2:
                        missingCount = log2[dataPointName]
                    missingCount = missingCount + 1
                    log2[dataPointName] = missingCount
    
    print("--- GetAllFinancialReportsDataPoints: %f minutes ---/n" % float((time.time() - start_time) / 60.0))


#a = {}
#log1 = {}
#log2 = {}
#GetAllFinancialReportsDataPoints(a, log1, log2)

def GetAllShortInterestRatios(allShortInterestRatios):

    print("--- Start GetAllShortInterestRatios ---")
    
    shortInterestRatios_Dir = ut.GetShortInterestRatioDir()
    shortInterestRatiosFile = shortInterestRatios_Dir + "\\Short Interest Ratios.xlsm"
    book = pd.ExcelFile(shortInterestRatiosFile)
    
    # All information is in one workbook
    sheet_Dates = book.parse("Dates")
    sheet_Values = book.parse("Values")
    sheet_ReportReleaseDates = book.parse("Eearings Release Dates")
    
    dict_Dates = sheet_Dates.to_dict()
    dict_Values = sheet_Values.to_dict()
    dict_ReportReleaseDates = sheet_ReportReleaseDates.to_dict()
    
    for companyName, companyReleaseDates in dict_ReportReleaseDates.items():
        shortInterestDates = list(dict_Dates[companyName].values())
        shortInterestRatios = list(dict_Values[companyName].values())
        
        for quarterName, quarterReleaseDate in companyReleaseDates.items():
            # For all the quarters that have a valid Report Release Date, find the nearest Short Interest Ratio before this Report Release Date
            if math.isnan(quarterReleaseDate) == False:
                for i in range(0, len(shortInterestDates)):
                    if shortInterestDates[i] > quarterReleaseDate and i > 0:
                        shortInterestIdx = i -1
                        shortInterestRatio = shortInterestRatios[shortInterestIdx]
                        if math.isnan(shortInterestRatio) == False:
                            quarterName = ut.transformQuarterName(quarterName)
                            allShortInterestRatios[(companyName, quarterName)] = shortInterestRatio
                            break
                        else:
                            shortInterestIdx = i -2
                            if shortInterestIdx >= 0:
                                shortInterestRatio = shortInterestRatios[shortInterestIdx]
                                if math.isnan(shortInterestRatio) == False:
                                    quarterName = ut.transformQuarterName(quarterName)
                                    allShortInterestRatios[(companyName, quarterName)] = shortInterestRatio
                                    break
                                
#allShortInterestRatios = {}
#GetAllShortInterestRatios(allShortInterestRatios)                              
                                    
                                
                                
                                
                                
                                
#a = trainingData_OneScenario['Training Points']                                
#aapl = a['AAPL']                                
#stockReturns = pd.DataFrame()  
#stockReturns = aapl['Outputs_Regression']                              
#toNormalise = []                                
#for i in range(len(stockReturns)):
#    singleReturn = stockReturns[i]
#    toNormalise.append([singleReturn])                            
#                                
#scaler = preprocessing.MinMaxScaler().fit(toNormalise)                                
#stockReturns_normalized = scaler.transform(toNormalise)                                
# 
#stockReturns_normalized_df = pd.DataFrame(data = stockReturns_normalized,
#                                                index = aapl.index,
#                                                columns = ['Outputs_Regression'])                               
#aapl2 = aapl
#aapl2['Outputs_Regression'] = stockReturns_normalized_df   








                         
                                
#                                
#                                
#allReportDataPoints = {}
#log1={}
#log2={}                          
#                                
#start_time = time.time()
#    
#quarterList = ["1996 FQ4",	"1997 FQ1",	"1997 FQ2",	"1997 FQ3",	"1997 FQ4",	"1998 FQ1",	"1998 FQ2",	"1998 FQ3",	"1998 FQ4",	"1999 FQ1",	"1999 FQ2",	"1999 FQ3",	"1999 FQ4",	"2000 FQ1",	"2000 FQ2",	"2000 FQ3",	"2000 FQ4",	"2001 FQ1",	"2001 FQ2",	"2001 FQ3",	"2001 FQ4",	"2002 FQ1",	"2002 FQ2",	"2002 FQ3",	"2002 FQ4",	"2003 FQ1",	"2003 FQ2",	"2003 FQ3",	"2003 FQ4",	"2004 FQ1",	"2004 FQ2",	"2004 FQ3",	"2004 FQ4",	"2005 FQ1",	"2005 FQ2",	"2005 FQ3",	"2005 FQ4",	"2006 FQ1",	"2006 FQ2",	"2006 FQ3",	"2006 FQ4",	"2007 FQ1",	"2007 FQ2",	"2007 FQ3",	"2007 FQ4",	"2008 FQ1",	"2008 FQ2",	"2008 FQ3",	"2008 FQ4",	"2009 FQ1",	"2009 FQ2",	"2009 FQ3",	"2009 FQ4",	"2010 FQ1",	"2010 FQ2",	"2010 FQ3",	"2010 FQ4",	"2011 FQ1",	"2011 FQ2",	"2011 FQ3",	"2011 FQ4",	"2012 FQ1",	"2012 FQ2",	"2012 FQ3",	"2012 FQ4",	"2013 FQ1",	"2013 FQ2",	"2013 FQ3",	"2013 FQ4",	"2014 FQ1",	"2014 FQ2",	"2014 FQ3",	"2014 FQ4",	"2015 FQ1",	"2015 FQ2",	"2015 FQ3",	"2015 FQ4",	"2016 FQ1",	"2016 FQ2",	"2016 FQ3",	"2016 FQ4",	"2017 FQ1",	"2017 FQ2",	"2017 FQ3",	"2017 FQ4",	"2018 FQ1",	"2018 FQ2",	"2018 FQ3",	"2018 FQ4"]
#
#badStocks = ['MANH', 'MAS', 'BBT', 'JH', 'CMD', 'CCK']  # These are stock codes that have been found to have bad download from Bloomberg
#
#financialReportMetrics_ByCompany = {}
#metricNames = []
#
#import os
#testFolder = ut.GetFinancialReportMetricsDir()
#files = os.listdir(testFolder)
#
#for fileName in files:
#    
#    # Unpack the current file
#    metricName = fileName.replace(r"Static - ", "")
#    metricName = metricName.replace(r".xlsm", "")
#    metricNames.append(metricName)
#    
#    featureFile = testFolder + "\\" + fileName
#    book = pd.ExcelFile(featureFile)
#    sheetName = book.sheet_names[0]
#    sheet = book.parse(sheetName)          
#    
#    # Pre-processing
#    sheet = reports.WorksheetPreProcessing(sheet)
#    
#    # Structure of 'sheetDictionary' : Quarter name (key) -> company name (key) -> metric value of this company
#    sheetDictionary = sheet.to_dict()
#    
#    # Structure of 'financialReportMetrics_ByCompany' : company name (key) -> metric Name (key) -> metric value dictionary with quarter name as key and metric value as value
#    for quarterName, metricValues_AllCompanies in sheetDictionary.items():
#        quarterName = ut.transformQuarterName(quarterName)
#        
#        for companyName, metricValue in metricValues_AllCompanies.items():
#            # Do not include any bad stocks
#            if companyName not in badStocks:
#                allMetricValues_OneCompany = {}
#                if companyName in financialReportMetrics_ByCompany:
#                    allMetricValues_OneCompany = financialReportMetrics_ByCompany[companyName]
#                    
#                singleMetricValues_OneCompany = {}
#                if metricName in allMetricValues_OneCompany:
#                    singleMetricValues_OneCompany = allMetricValues_OneCompany[metricName]
#                
#                singleMetricValues_OneCompany[quarterName] = metricValue
#                allMetricValues_OneCompany[metricName] = singleMetricValues_OneCompany
#                financialReportMetrics_ByCompany[companyName] = allMetricValues_OneCompany
#
## Calculate quarterly and yearly diffs
#quarterlyDiffs = reports.calculateQuarterlyDiffs(financialReportMetrics_ByCompany, quarterList)
#yearlyDiffs = reports.calculateYearlyDiffs(financialReportMetrics_ByCompany, quarterList)
#
## Merge quarterly and yearly diff results AND native results of all ratio features
#ratioFeatures  = ['Current Ratio', 'Dividend Payout Ratio', 'Dividend Yield', 'Inventory Turnover', 'Net Debt to EBIT', 'Operating Margin', 'PB Ratios', 'PC Ratios', 'PE Ratios', 'PS Ratios', 'Quick Ratio', 'Return On Assets', 'Return On Common Equity', 'Total Debt to Total Assets', 'Total Debt to Total Equity']
#financialReportMetrics_ByCompany_Final = {}
#for companyName, quarterlyCompanyMetrics in quarterlyDiffs.items():
#    temp = deepcopy(quarterlyCompanyMetrics)
#    
#    yearlyCompanyMetrics = yearlyDiffs[companyName]    
#    temp.update(yearlyCompanyMetrics)
#    
#    nativeCompanyMetrics = financialReportMetrics_ByCompany[companyName]
#    nativeCompanyMetrics_new = {}
#    for key in ratioFeatures:
#        if key in nativeCompanyMetrics:
#            nativeCompanyMetrics_new[key] = nativeCompanyMetrics[key]
#    temp.update(nativeCompanyMetrics_new)
#    
#    financialReportMetrics_ByCompany_Final[companyName] = temp
#
## Re-organize things by data points of (companyName, quarterName)
#for companyName, companyMetrics in financialReportMetrics_ByCompany_Final.items():
#    for metricName, metricValues in companyMetrics.items():
#        for quarterName, metricValue in metricValues.items():
#            
#            pointName = (companyName, quarterName)
#            
#            # PointValue is a dictionary of numerous metrics for the current (companyName, quarterName)
#            pointValue = {}
#            if pointName in allReportDataPoints:
#                pointValue = allReportDataPoints[pointName]
#                
#            if str(type(metricValue)) == "<class 'str'>":
#                pointValue[metricName] = 0  # Set cells of 'string' type to 0, effectively eliminating all NaNs and empty strings
#            else:
#                pointValue[metricName] = metricValue
#                
#            allReportDataPoints[pointName] = pointValue
#            
## Add in each data point's missing metrics and set the value of phantom metrics to 0
#myChosenFeatures = metricNames
#tempQ = [i + "_Q_Change" for i in myChosenFeatures]
#tempY = [i + "_Y_Change" for i in myChosenFeatures]
#
## Chosen features include Quarterly changes and Yearly changes of ALL features, AND the ratio features
#myChosenFeatures = tempQ + tempY + ratioFeatures  
#
## Diagnosis - find out what is missing
#for dataPointName, dataPointValue in allReportDataPoints.items():            
#    if len(dataPointValue) != len(myChosenFeatures):
#        for chosenFeature in myChosenFeatures:
#            if chosenFeature not in dataPointValue:
#                dataPointValue[chosenFeature] = 0
#                
#                # Logging
#                missingMetrics = []
#                if dataPointName in log1:
#                    missingMetrics = log1[dataPointName]
#                missingMetrics.append(chosenFeature)
#                log1[dataPointName] = missingMetrics
#                
#                missingCount = 0
#                if dataPointName in log2:
#                    missingCount = log2[dataPointName]
#                missingCount = missingCount + 1
#                log2[dataPointName] = missingCount        
                                    
                                
                                
                                