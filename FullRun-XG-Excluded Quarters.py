'''

Created on 21/02/2020

Once created, this file must not be changed.

'''


import pandas as pd
from copy import deepcopy
import numpy as np
import HyperparameterTuning_XG as HT
from Utilities import computeSampleWeights
import time
import xgboost as xgb
import PrepareTrainingData as data
import Utilities as ut
import pandas as pd
import time
import datetime
from copy import deepcopy
from operator import itemgetter
start_time = time.time()

#==============Controls==============
includeMinus1Dto1D = False
testSpecificDate = False
doCAR = False
useMarketCap = ""
useMinMaxScalingForInputs = False
TestType = "Classification" #"Classification" "Regression"
scalingMethodForPrices = ""     #"ZSpread"
scenario = {"start" : "-1D", "end" : "30D"}  # For example, if 'start' is 1D, it means we start forecasting after the close of 1D having known the 1D prices
use_Data_From_Year = 2003  #2010
exclude2008Crsis = True
#==============Controls==============

#==============Controls==============

inProcessFeatureSelection = False

performModelTuning = True

doSampleWeight = True

classificationMetric = 'error'

validationSetPercentage = 0.1

#
populationCount_HT = 10
generationCount_HT = 3
#

priorFeatureSelection = False
#remaining_columns = ["priceReaction_minus1D_to_1D"]
#remaining_columns = [i for i in x_train.columns if i not in remaining_columns]

ITERATIONS = 100
#==============Controls==============

result_index = ['sector', 'year', 'test count', 'idx', 'rate','eta', 'max_depth', 'colsample_bytree', 'subsample', 'gamma', 'min_child_weight','a1','a2','a3','a4','a5','a6','a7','a8','a9','a10','b1','b2','b3','b4','b5','b6','b7','b8','b9','b10']
a_final_results = pd.DataFrame(columns=result_index)
'''
1. Data Loading and Feature Generation
'''

Regenerate_Features = False

if Regenerate_Features == True:
    
    #'''
    #1. Financial Reports Data
    #'''
    allReportDataPoints = {}
    log1={}
    log2={}
    data.GetAllFinancialReportsDataPoints(allReportDataPoints, log1, log2)
    
    #'''
    #3. Stock prices data
    #'''
    priceReactions = {} 
    priceReactionForward = {}
    
    data.GetALLStockPriceData(scenario, priceReactions, priceReactionForward, doCAR, includeMinus1Dto1D)
    #
    #'''
    #4. Technical Data
    #'''
    allTechnicalData_TupleFormat = {}
    data.GetALLTechnicalData(allTechnicalData_TupleFormat, scenario)
    #
    #'''
    #6. Sector Data
    #'''
    allSectorData = {}
    data.GetAllSectorData(allSectorData)
    
    #'''
    #7. Earnings Data
    #'''
    allEarningsData = {}
    data.GetAllEarningsData(allEarningsData)
    
    #'''
    #8. Short Interest Ratio Data
    #'''
    allShortInterestRatios = {}
    data.GetAllShortInterestRatios(allShortInterestRatios)
    
    #'''
    #9. Earning release date
    #'''
    releaseDateData = {}
    data.GetReleaseDates(releaseDateData)   
    
    #'''
    #10. Market cap
    #'''
    marketCaps = {}
    data.GetMarketCaps(marketCaps)   
    
    
    #'''
    #9. Form final set of training data (final inputs and outputs of learning models)
    #'''
    trainingData_OneScenario = {}
    trainingData_OneScenario = data.GetTrainingData_OneScenario( 
                                                           allReportDataPoints, 
                                                           allTechnicalData_TupleFormat, 
                                                           allShortInterestRatios,
                                                           allSectorData,                                                        
                                                           allEarningsData,
                                                           releaseDateData,
                                                           marketCaps,
                                                           priceReactions, 
                                                           priceReactionForward,
                                                           use_Data_From_Year,
                                                           exclude2008Crsis)
    
    # Company level pre-processing
    priceScalers = {}
    trainingData_OneScenario_processed = data.CompanyLevelPreProcessing(trainingData_OneScenario, allReportDataPoints, allSectorData, useMinMaxScalingForInputs, scalingMethodForPrices, priceScalers)
    
    # Pack all company data together
    trainingData_OneScenario_processed_flat = data.FlattenTrainingData(trainingData_OneScenario_processed)


'''
2. Loop through all sectors
'''
Test_Quarters_Set = {'2018' : ['2018 FQ4', '2018 FQ3', '2018 FQ2', '2018 FQ1'], 
                     '2017' : ['2017 FQ4', '2017 FQ3', '2017 FQ2', '2017 FQ1'], 
                     '2016' : ['2016 FQ4', '2016 FQ3', '2016 FQ2', '2016 FQ1'], 
                     '2015' : ['2015 FQ4', '2015 FQ3', '2015 FQ2', '2015 FQ1'], 
                     '2014' : ['2014 FQ4', '2014 FQ3', '2014 FQ2', '2014 FQ1']}

Years = ['2014']
#Years = ['2018', '2017', '2016', '2015', '2014']

Sectors = ['all']
#Sectors = ['Energy', 'Financial', 'Industrial', 'Technology', 'Utilities', 'Basic Materials', 'Communications', 'Consumer, Cyclical', 'Consumer, Non-cyclical']
#Sectors = ['all', 'Energy', 'Financial', 'Industrial', 'Technology', 'Utilities', 'Basic Materials', 'Communications', 'Consumer, Cyclical', 'Consumer, Non-cyclical']

#Years = ['2018', '2017']
#Sectors = ['Consumer, Cyclical', 'Consumer, Non-cyclical']

run_count = 1
for sector in Sectors:
    
    trainingData = trainingData_OneScenario_processed_flat.get(sector)
    
    print("Index = %i, Sector = %s" % (run_count, sector))
    print("--- Start GenerateXY ---")
    start_time = time.time()
    
    # Filter rows according to 'Market Cap'
    if "Market Cap" in trainingData:
        if len(useMarketCap) != 0:
            trainingData = trainingData.loc[trainingData['Market Cap'] == useMarketCap]
            
        # Drop the Market Cap column
        trainingData = trainingData.drop(["Market Cap"], axis = 1)
        
    '''
    3. Loop through each year
    '''    
    
    for year in Years:
    
        print("Index = %i, Sector = %s, Year = %s" % (run_count, sector, year))
        
        '''
        4. Divide training data into training set and test set
        ''' 
        test_quarters = Test_Quarters_Set[year]
        
        featureCount = int()    
        x_train = pd.DataFrame()
        y_train = pd.DataFrame() 
        x_test = pd.DataFrame()
        y_test = pd.DataFrame()    
        trainingSet = pd.DataFrame()
        testingSet = pd.DataFrame()
        
        for index, row in trainingData.iterrows():
            company = index[0]
            quarter = index[1]
            
            Ignore_Quarters = []

            if quarter in test_quarters:
                testingSet = testingSet.append(row)
            elif quarter in Ignore_Quarters:
                a = 1
            else:
                trainingSet = trainingSet.append(row)
        
        # keep the column orders                
        testingSet = testingSet[trainingData.columns]   
        trainingSet = trainingSet[trainingData.columns]
            
        # Drop the 'Release Date' column
        trainingSet = trainingSet.drop(["Release Date"], axis = 1)
        testingSet = testingSet.drop(["Release Date"], axis = 1)
         
        from sklearn.utils import shuffle
        testingSet = shuffle(testingSet)
        trainingSet = shuffle(trainingSet)
        
        x_train = deepcopy(trainingSet.iloc[:, :-2])
        x_test = deepcopy(testingSet.iloc[:, :-2])
        
        if TestType == "Regression":
            y_train = deepcopy(trainingSet['Outputs_Regression'])        
            y_test = deepcopy(testingSet['Outputs_Regression'])
        else:
            y_train = deepcopy(trainingSet['Outputs_Classification'])        
            y_test = deepcopy(testingSet['Outputs_Classification'])                
                    
        featureNames = x_train.columns
        featureCount = len(featureNames)
        print("--- GenerateXY: %f minutes ---/n" % float((time.time() - start_time) / 60.0))

        '''
        5. Start training and testing
        ''' 
        for iterationIdx in range(ITERATIONS):
            
            print("Index = %i, Sector = %s, Year = %s, Iteration = %i" % (run_count, sector, year, iterationIdx))
            
                
            '''
            5.1. +++++ Feature Selection +++++
            '''
            
            x_train_FeatureSelected = []
            x_test_FeatureSelected = []
            if priorFeatureSelection == False:
                x_train_FeatureSelected = deepcopy(x_train)
                x_test_FeatureSelected = deepcopy(x_test)
            else:    
                
                x_train_FeatureSelected = deepcopy(x_train)
                x_test_FeatureSelected = deepcopy(x_test)
                
                x_train_FeatureSelected = x_train.filter(remaining_columns)
                x_test_FeatureSelected = x_test.filter(remaining_columns)
                
            
            '''
            5.2. +++++ Separate train, validation, and test data +++++
            '''
            # Instantiations
            n_folds = 5
            early_stopping = 10
            n_estimators = 4000
            
            maxDepth_in = int()
            subSampleRate_in = float()
            colSampleRate_in = float()
            gamma_in = float()
            learningRate_in = float()
            min_child_weight_in = float()
            bestFeaturesFromTuning = []
            
            # Use the first 10% of training data points as validation set to achieve the best 'num_boost_round', helped by 'early_stopping_rounds'
            x_train_FeatureSelected.rename(columns = lambda x: x.replace(' ', '_'), inplace=True)
            x_test_FeatureSelected.rename(columns = lambda x: x.replace(' ', '_'), inplace=True)
            
            validSetSize = int(len(x_train_FeatureSelected) * validationSetPercentage)
            
            x_train_val = x_train_FeatureSelected[:validSetSize]
            y_train_val = y_train[:validSetSize]
            
            x_train_final = x_train_FeatureSelected[(validSetSize+1):]
            y_train_final = y_train[(validSetSize+1):]
                
            '''
            5.3. +++++ Compute weights of imbalanced data - when performing Classification +++++
            '''
            classCount = len(np.unique(y_train))
            sample_weight = computeSampleWeights(y_train, TestType, doSampleWeight)
            sample_weight_val = sample_weight[:validSetSize]
            sample_weight_training = sample_weight[(validSetSize+1):]
            
            '''
            5.4. +++++ Start tuning +++++
            '''
            
            # Tune model
            if performModelTuning == True:
                
                bestParams_All_1 = []
                minFitnessInTuning = []
                allFitnessLog = []
                finalPopulation = []
                
                ret = HT.PerformHyperparameterTuning(TestType,
                                                     x_train_final, 
                                                     y_train_final, 
                                                     sample_weight_training,
                                                     n_estimators,
                                                     n_folds,
                                                     early_stopping,
                                                     inProcessFeatureSelection,
                                                     classificationMetric,
                                                     populationCount_HT, 
                                                     generationCount_HT)
                
                bestParams_All_1.append(ret[0] )
                minFitnessInTuning = ret[1]
                allFitnessLog = ret[2]
                finalPopulation = ret[3]
                
                # Extract the tuned parameters for later use
                bestParams = ret[0]
                
                maxDepth_in = int(bestParams[0])
                subSampleRate_in = bestParams[1]
                colSampleRate_in = bestParams[2]
                gamma_in = bestParams[3]
                learningRate_in = bestParams[4]
                min_child_weight_in = bestParams[5]
                bestFeaturesFromTuning = bestParams[6]
                
                
            print("\nGA results:eta = %f, max_depth = %i, colsample_bytree = %f, subsample = %f, gamma = %f, min_child_weight = %i" % (learningRate_in, maxDepth_in, colSampleRate_in, subSampleRate_in, gamma_in, min_child_weight_in) )    
            
                 
                
            '''
            5.4. +++++ Train model using tuned params, then Predict +++++
            '''
            classification_accuracy = 0
            if TestType == "Regression":
                
                '''
                a. ======== Prepare for all training parameters ========
                '''
                objective_in = 'reg:linear'
                booster_in = 'gbtree'
                METRICS = ['mae', 'rmse']
                
                '''
                b. ======== Apply tuned parameters to train model and perform prediction based on test dat ========
                '''
                
                xgb_reg = xgb.XGBRegressor(max_depth = maxDepth_in,          # set
                                           learning_rate = learningRate_in,  # set 
                                           n_estimators = n_estimators,               # set as hardcoded number
                                           silent=True, 
                                           objective = objective_in,       # set  
                                           booster = booster_in,           # set 
                                           n_jobs=4, 
                                           gamma = gamma_in,                 # set
                                           min_child_weight = min_child_weight_in,
                                           max_delta_step=0, 
                                           subsample = subSampleRate_in,   # set
                                           colsample_bytree = colSampleRate_in,    # set 
                                           colsample_bylevel=1, 
                                           reg_alpha=0, 
                                           reg_lambda=1, 
                                           scale_pos_weight=1, 
                                           base_score=0.5, 
                                           random_state=0, 
                                           seed=None, 
                                           missing=None, 
                                           importance_type='gain')
                
                xgb_reg.fit(x_train_final, 
                            y_train_final, 
                            eval_set=[(x_train_val, y_train_val)], 
                            eval_metric=METRICS,
                            early_stopping_rounds = early_stopping,
                            verbose=False)
                
                '''
                c. ======== Prediction and Result Evaluation ========
                '''
                
                # Predict
                y_predicted = xgb_reg.predict(x_test_FeatureSelected)
                
                y_predicted_old = deepcopy(y_predicted)
                y_test_old = deepcopy(y_test)
                
                if scalingMethodForPrices != "":
                    for i in range(len(y_test_old)):
                        companyName = y_test_old.index[i][0]
                        #print(companyName)
                        scaler = priceScalers[companyName]
                        #print(y_test[i])
                        testValue = scaler.inverse_transform( [[y_test_old[i]]])
                        #print(testValue)
                        predictedValue = scaler.inverse_transform([[y_predicted_old[i]]])
                        y_test[i] = testValue[0][0]
                        y_predicted[i] = predictedValue[0][0]
                        
                
                print("\nUnique:")
                print(len(np.unique(y_predicted)))
                   
                from sklearn.metrics import mean_absolute_error, mean_squared_error
                mae = mean_absolute_error(y_test, y_predicted)
                rmse = mean_squared_error(y_test, y_predicted)
                print("\nmae: %.3f%%, rmse: %.3f%% \n" % (100*mae, 100*rmse))
                
            else:
                
                '''
                a. ======== Prepare for all training parameters ========
                '''
                # Define objective and metric types
                if classCount == 2:
                    objective = 'binary:logistic'
                    #metrics = ['aucpr', 'map', 'error']
                    metrics = [classificationMetric]
                else:
                    objective = 'multi:softprob'
                    metrics = ['merror']
                    
                # Complete params
                params = {'eta': learningRate_in, 
                          'max_depth': maxDepth_in, 
                          'colsample_bytree': colSampleRate_in, 
                          'subsample': subSampleRate_in,               
                          'gamma':gamma_in,              
                          'min_child_weight':min_child_weight_in,
                          'eval_metric': metrics,     
                          'silent': 1, 
                          'nthread':4,
                          'objective': objective}
                
                if classCount > 2:
                    params['num_class'] = classCount
                        
                    
                '''
                b. ======== Apply tuned parameters to train model and perform prediction based on test dat ========
                '''
                
                traindata_matrix = xgb.DMatrix(x_train_final, label = y_train_final, weight = sample_weight_training)
                valdata_matrix = xgb.DMatrix(x_train_val, label = y_train_val, weight = sample_weight_val) 
                
                print("\nEvaluation stats during actual model training:")
                evals_result = {}
                #watchlist = [(valdata_matrix, "eval")]
                watchlist = [(traindata_matrix, "train"),(valdata_matrix, "eval")]
                
                # early_stopping_rounds can be set in this method
                xg_trained = xgb.train(params, 
                                       traindata_matrix, 
                                       early_stopping_rounds = 100,#early_stopping,
                                       num_boost_round = n_estimators, 
                                       verbose_eval = False,
                                       evals = watchlist, 
                                       evals_result = evals_result)
                
                    
                '''
                c. ======== Prediction and Result Evaluation ========
                '''
                
                # Predict
                y_predicted = xg_trained.predict(xgb.DMatrix(x_test_FeatureSelected))
                
                # Calculate result metrics
                # https://scikit-learn.org/stable/modules/model_evaluation.html
                
                # 'accuracy_score, precision_score, recall_score and  are methods for classification only
                from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
                
                y_predicted_MultiClass = []
                if classCount == 2:
                    y_predicted_MultiClass = [int(round(i)) for i in y_predicted]
                else:
                    y_predicted_MultiClass = np.asarray([np.argmax(i) for i in y_predicted])
                        
                classification_accuracy = accuracy_score(y_test, np.array(y_predicted_MultiClass))            
                print("\nAccuracy: %.2f%%\n" % (classification_accuracy * 100.0))
            
                precision = precision_score(y_test, y_predicted_MultiClass,average='micro')     # true positive / (true positive + false positive)
                recall = recall_score(y_test, y_predicted_MultiClass, average='micro')          # true positive / (true positive + false negative)
                print("--precision = %f, recall = %f\n" % (precision, recall))
                
                if classCount == 2:
                    precision = precision_score(y_test, y_predicted_MultiClass,average='micro')     # true positive / (true positive + false positive)
                    recall = recall_score(y_test, y_predicted_MultiClass, average='micro')          # true positive / (true positive + false negative)
                    print("--precision = %f, recall = %f\n" % (precision, recall))
            
                target_names = ['class ' + str(i) for i in np.unique(y_train)]
                print(classification_report(y_test, y_predicted_MultiClass, target_names=target_names))
            
            '''
            6. Save result as a row of the result dataframe
            ''' 
            row_result = list()
            row_result.append(sector)
            row_result.append(year)
            row_result.append(len(y_test))
            row_result.append(iterationIdx)
            row_result.append(classification_accuracy)
            row_result.append(learningRate_in)
            row_result.append(maxDepth_in)
            row_result.append(colSampleRate_in)
            row_result.append(subSampleRate_in)
            row_result.append(gamma_in)
            row_result.append(min_child_weight_in)
            
            key_features = xg_trained.get_score(importance_type = 'gain')
            sorted_key_features =sorted(key_features.items(), key=itemgetter(1), reverse=True)
            
            usedFeatureCount = 10
            paddingSize = 0
            if usedFeatureCount > len(sorted_key_features):
                usedFeatureCount = len(sorted_key_features)
                paddingSize = 10 - usedFeatureCount
                
            for i in range(usedFeatureCount):
                row_result.append(sorted_key_features[i][0])
            
            if paddingSize > 0:
                for i in range(paddingSize):
                    row_result.append("")
                    
            for i in range(usedFeatureCount):
                row_result.append(sorted_key_features[i][1])
            
            if paddingSize > 0:
                for i in range(paddingSize):
                    row_result.append("")
                    
            row_result_series = pd.Series(row_result, index = result_index)
            a_final_results = a_final_results.append(row_result_series, ignore_index=True)
            run_count = run_count + 1
    
    

