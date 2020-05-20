
import PrepareTrainingData as data
import Utilities as ut
import pandas as pd
import time
import datetime
from copy import deepcopy

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
#==============Controls==============



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
                                                       use_Data_From_Year)

# Company level pre-processing
priceScalers = {}
trainingData_OneScenario_processed = data.CompanyLevelPreProcessing(trainingData_OneScenario, allReportDataPoints, allSectorData, useMinMaxScalingForInputs, scalingMethodForPrices, priceScalers)

# Pack all company data together
trainingData_OneScenario_processed_flat = data.FlattenTrainingData(trainingData_OneScenario_processed)
#trainingData_OneScenario_unprocessed_flat = data.FlattenTrainingData(trainingData_OneScenario)

'''
9. Finalise the training data
'''     
# all, Energy, Financial, Industrial, Technology, Utilities, Basic Materials, Communications, "Consumer, Cyclical", "Consumer, Non-cyclical"
#sectors = ["Financial"]      
sectors = ["Financial"]
trainingData = pd.concat([trainingData_OneScenario_processed_flat.get(sector) for sector in sectors], axis=0)

    
'''
10. Divide training data into training set and test set
'''    

print("--- Start GenerateXY ---")
start_time = time.time()

# Filter rows according to 'Market Cap'
if "Market Cap" in trainingData:
    if len(useMarketCap) != 0:
        trainingData = trainingData.loc[trainingData['Market Cap'] == useMarketCap]
        
    # Drop the Market Cap column
    trainingData = trainingData.drop(["Market Cap"], axis = 1)

featureCount = int()    
x_train = pd.DataFrame()
y_train = pd.DataFrame() 
x_test = pd.DataFrame()
y_test = pd.DataFrame()
trainingSet = pd.DataFrame()
testingSet = pd.DataFrame()

if testSpecificDate == True:
    for index, row in trainingData.iterrows():
        releaseDate = row[-1]
        if releaseDate == datetime.datetime(2018, 10, 25, 0, 0):
#        releaseDate == datetime.datetime(2018, 4, 26, 0, 0):
#        or releaseDate == datetime.datetime(2018, 10, 30, 0, 0) \
#        or releaseDate == datetime.datetime(2018, 10, 31, 0, 0) \
#        or releaseDate == datetime.datetime(2018, 11, 1, 0, 0) \
#        or releaseDate == datetime.datetime(2018, 11, 2, 0, 0) :
            testingSet = testingSet.append(row)
        else:
            trainingSet = trainingSet.append(row)
            
else:
    
    # Keep those data points whose EPS EarningsSurprise is >= 0
    #    trainingData = trainingData[trainingData['EPS EarningsSurprise'] >= 0]

    for index, row in trainingData.iterrows():
        company = index[0]
        quarter = index[1]
        
#        Test_Set_Quarters = ['2018 FQ2']
#        Ignore_Quarters = []
        
        Test_Set_Quarters = ['2016 FQ4',
                             '2016 FQ3',
                             '2016 FQ2',
                             '2016 FQ1']
        Ignore_Quarters = []
#        Ignore_Quarters = ['2017 FQ4',
#                             '2017 FQ3',
#                             '2017 FQ2',
#                             '2017 FQ1',
#                             '2016 FQ4',
#                             '2016 FQ3',
#                             '2016 FQ2',
#                             '2016 FQ1']
        
        if quarter in Test_Set_Quarters:
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





#
#if testSpecificDate == True:
#    
#    # Separate traing data into traing set and testing set
#    trainingSet = pd.DataFrame()
#    testingSet = pd.DataFrame()
#    for index, row in trainingData.iterrows():
#        releaseDate = row[-1]
#        if releaseDate == datetime.datetime(2018, 10, 25, 0, 0):
##        releaseDate == datetime.datetime(2018, 4, 26, 0, 0):
##        or releaseDate == datetime.datetime(2018, 10, 30, 0, 0) \
##        or releaseDate == datetime.datetime(2018, 10, 31, 0, 0) \
##        or releaseDate == datetime.datetime(2018, 11, 1, 0, 0) \
##        or releaseDate == datetime.datetime(2018, 11, 2, 0, 0) :
#            testingSet = testingSet.append(row)
#        else:
#            trainingSet = trainingSet.append(row)
#
#    testingSet = testingSet[trainingData.columns]   # keep the column orders
#    trainingSet = trainingSet[trainingData.columns]
#        
#    # Drop the Release Date column
#    trainingSet = trainingSet.drop(["Release Date"], axis = 1)
#    testingSet = testingSet.drop(["Release Date"], axis = 1)
#     
#    from sklearn.utils import shuffle
#    testingSet = shuffle(testingSet)
#    trainingSet = shuffle(trainingSet)
#    
#    x_train = deepcopy(trainingSet.iloc[:, :-2])
#    x_test = deepcopy(testingSet.iloc[:, :-2])
#
#    if TestType == "Regression":
#        y_train = deepcopy(trainingSet['Outputs_Regression'])        
#        y_test = deepcopy(testingSet['Outputs_Regression'])
#    else:
#        y_train = deepcopy(trainingSet['Outputs_Classification'])        
#        y_test = deepcopy(testingSet['Outputs_Classification'])
#    
#    
#    
#else:
#
#    testSpecificQuarters = True
#    
#    # Drop the Release Date column
#    #if "Release Date" in trainingData:
#    #    trainingData = trainingData.drop(["Release Date"], axis = 1)
#          
#    if testSpecificQuarters == True:
#        
#        # Keep those data points whose EPS EarningsSurprise is >= 0
#    #    trainingData = trainingData[trainingData['EPS EarningsSurprise'] >= 0]
#        
#        # Separate traing data into traing set and testing set
#        trainingSet = pd.DataFrame()
#        testingSet = pd.DataFrame()
#        for index, row in trainingData.iterrows():
#            company = index[0]
#            quarter = index[1]
#            if quarter == '2018 FQ4':
#    #        or quarter == '2018 FQ1' \
#    #        or quarter == '2017 FQ4' \
#    #        or quarter == '2017 FQ3' \
#    #        or quarter == '2017 FQ2' \
#    #        or quarter == '2017 FQ1':
#    
##                a=1
##            elif quarter == '2018 FQ3':
#                
#                testingSet = testingSet.append(row)
#            else:
#                trainingSet = trainingSet.append(row)
#    
#        testingSet = testingSet[trainingData.columns]   # keep the column orders
#        trainingSet = trainingSet[trainingData.columns]
#            
#        # Drop the Release Date column
#        trainingSet = trainingSet.drop(["Release Date"], axis = 1)
#        testingSet = testingSet.drop(["Release Date"], axis = 1)
#    
#        from sklearn.utils import shuffle
#        testingSet = shuffle(testingSet)
#        trainingSet = shuffle(trainingSet)
#        
#        x_train = deepcopy(trainingSet.iloc[:, :-2])
#        x_test = deepcopy(testingSet.iloc[:, :-2])
#
#        if TestType == "Regression":
#            y_train = deepcopy(trainingSet['Outputs_Regression'])        
#            y_test = deepcopy(testingSet['Outputs_Regression'])
#        else:
#            y_train = deepcopy(trainingSet['Outputs_Classification'])        
#            y_test = deepcopy(testingSet['Outputs_Classification'])
#        
#        
#    else:
#        
##        inputPoints = trainingData.iloc[:, :-2]
##        outputPointsRegress = trainingData['Outputs_Regression']
##        outputPointsClass = trainingData['Outputs_Classification']
##        
##        outputPoints = pd.DataFrame()
##        
##        if TestType == "Regression":
##            outputPoints = outputPointsRegress  
##        else:
##            outputPoints = outputPointsClass 
##        #trainingData = trainingData.loc[trainingData["Outputs_Regression"] < 0.05]
##        
##        # Split the test data into training set and test set. Test set is not involved in the GA process.
##        from sklearn.model_selection import train_test_split
##        import random
##        x_train, x_test, y_train, y_test = train_test_split(inputPoints,
##                                                            outputPoints,
##                                                            test_size = 0.2,
##                                                            random_state = random.randint(1,50))

#featureNames = x_train.columns
#featureCount = len(featureNames)








################################################################################################
  


#scenarios = []
#scenarios.append({"start" : "-1D", "end" : "0D"})
#scenarios.append({"start" : "-1D", "end" : "1D"})
#scenarios.append({"start" : "-1D", "end" : "10D"})
#scenarios.append({"start" : "-1D", "end" : "30D"})
#scenarios.append({"start" : "-1D", "end" : "60D"})
#scenarios.append({"start" : "-15D", "end" : "-1D"})
#scenarios.append({"start" : "1D", "end" : "10D"})
#scenarios.append({"start" : "1D", "end" : "30D"})
#scenarios.append({"start" : "1D", "end" : "60D"})
#scenarios.append({"start" : "2D", "end" : "10D"})
#scenarios.append({"start" : "2D", "end" : "30D"})
#scenarios.append({"start" : "2D", "end" : "60D"})
#scenarios.append({"start" : "3D", "end" : "10D"})
#scenarios.append({"start" : "3D", "end" : "30D"})
#scenarios.append({"start" : "3D", "end" : "60D"})
#priceForwards = {}
#Infos = {}
#for scenario in scenarios:
#    priceReactions = {} 
#    priceReactionForward = {}
#    info = {}
#    data.GetALLStockPriceData(scenario, priceReactions, priceReactionForward, info)
#    key = scenario["start"] + " " + scenario["end"]
#    priceForwards[key] = priceReactionForward
#    Infos[key] = info
#    


     

'''
The following code uses a very small subset of training data to evaluate
whether the model will quickly overfit the data
''' 
#        
#import numpy as np
#import FeatureSelection as FS
#import HyperparameterTuning-NN as HT
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#from keras.models import Sequential
#from keras.layers import Dense, Dropout
#from keras import regularizers
#from keras import optimizers
#import time
#
## Split the test data into training set and test set. Test set is not involved in the GA process.
#x_train, x_test, y_train, y_test = train_test_split(np.array([np.array(x) for x in inputPoints]),
#                                                    outputPoints,
#                                                    test_size = 0.2)
#
## For each fold, standardize the training data.           
#scalerX = StandardScaler()
#scalerX.fit(x_train)
#x_train = scalerX.transform(x_train.tolist())
#
## Also standardize the validation data using the same scaler.      
#x_test = scalerX.transform(x_test.tolist())
#
## Use the whole training set to re-train the model using the selected set of features and tuned hyperparameters
#epochs_count = 100
#hiddenLayerNeuronCount = 16
#batchSize = 512  
#learningRate = 0.01
#
#model = Sequential()
#model.add(Dense(hiddenLayerNeuronCount, input_dim = featureCount, activation='tanh'))
#model.add(Dense(hiddenLayerNeuronCount, activation='tanh'))
#model.add(Dense(hiddenLayerNeuronCount, activation='tanh'))
#model.add(Dense(1, activation='linear'))
#adam = optimizers.Adam(lr = learningRate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) #0.001
#model.compile(loss='mse', optimizer=adam, metrics = ['mae']) # calculate additional metric 'mean absolute error'
#
#fitResult = model.fit(x_train, 
#                       y_train, 
#                       epochs = epochs_count, 
#                       batch_size = batchSize, 
#                       verbose = 1, 
#                       validation_data=(x_test, y_test))
#
#a_val_loss = fitResult.history["val_loss"][-1]
#a_loss = fitResult.history["loss"][-1]
#a_mae = fitResult.history["mean_absolute_error"][-1]
#
#from keras import backend as K
#K.clear_session()





################################################################################################

'''
The following code performs feature selection and hyperparameter tuning based on neural network
''' 

#
#
#import numpy as np
#import FeatureSelection as FS
#import HyperparameterTuning as HT
#from sklearn.model_selection import train_test_split
#from keras.models import Sequential
#from keras.layers import Dense, Dropout
#from keras import regularizers
#import time
#
#
#
#start_time = time.time()
#featureCount = len(featureNames)
#batchSize = 128
#
## Split the test data into training set and test set. Test set is not involved in the GA process.
#x_train, x_test, y_train, y_test = train_test_split(np.array([np.array(x) for x in inputPoints]),
#                                                    outputPoints,
#                                                    test_size = 0.2)
#
### Feature selection
##populationCount_FS = len(featureNames)
##generationCount_FS = 20
##epochs_count = 10    
##hiddenLayerNeuronCount = 32
##ret = FS.PerformFeatureSelection(x_train, y_train, len(featureNames), batchSize, epochs_count, hiddenLayerNeuronCount, populationCount_FS, generationCount_FS)
##bestFeatures = ret[0]
##timeConsumed_FeatureSelection = (time.time() - start_time) / 60.0
##
### Limit features to only those selected
##x_train_FeatureSelected = []
##for point in x_train:
##    filtered = [i*j for i,j in zip(bestFeatures, point)]
##    x_train_FeatureSelected.append(np.array(filtered))
##x_train_FeatureSelected = np.array(x_train_FeatureSelected)
##y_train = np.array(y_train)
#
#
#x_train_FeatureSelected = x_train
#
#
## Hyperparameter tuning
#populationCount_HT = 20
#generationCount_HT = 20
#batchSize = 1028                    
#init_epochs_count = 5       
#init_hiddenLayerNeuronCount = 64         
#init_dropoutRate = 0.3 
#init_reg_lambda = 0.001
#ret = HT.PerformHyperparameterTuning(x_train_FeatureSelected, 
#                                     y_train, 
#                                     len(featureNames), 
#                                     batchSize, 
#                                     init_epochs_count, 
#                                     init_hiddenLayerNeuronCount, 
#                                     init_dropoutRate,
#                                     init_reg_lambda,
#                                     populationCount_HT, 
#                                     generationCount_HT)
#tunedParameters = ret[0]
#currentAverageFitness = ret[2]
#timeConsumed_HyperTuning = (time.time() - start_time) / 60.0 - timeConsumed_FeatureSelection
#
##==============================================#
## Test out-of-sample performance
##==============================================#
#
#
## Use the whole training set to re-train the model using the selected set of features and tuned hyperparameters
#epochs_count = int(tunedParameters[0])
#hiddenLayerNeuronCount = int(tunedParameters[1])
#dropoutRate = tunedParameters[2]
#reg_lambda = tunedParameters[3]
#
#model = Sequential()
#model.add(Dense(hiddenLayerNeuronCount, kernel_regularizer=regularizers.l2(reg_lambda), input_dim = featureCount, activation='relu'))
#model.add(Dropout(dropoutRate))
#model.add(Dense(hiddenLayerNeuronCount, kernel_regularizer=regularizers.l2(reg_lambda), activation='relu'))
#model.add(Dropout(dropoutRate))
#model.add(Dense(1, activation='linear'))
#model.compile(loss='mse', optimizer='adam', metrics = ['mae']) # calculate additional metric 'mean absolute error'
#
#result_TestSet = model.fit(x_train_FeatureSelected, y_train, epochs = epochs_count, batch_size = batchSize, verbose = 0)
#result_TestSet_history = result_TestSet.history
#
## Predict using the un-used test set data to which feature selection has been applied
##x_test_filtered_temp = []
##for point in x_train:
##    filtered = [i*j for i,j in zip(bestFeatures, point)]
##    x_test_filtered_temp.append(np.array(filtered))
##x_test_filtered = np.array(x_test_filtered_temp)
##y_test_predicted = model.predict(x_test_filtered, verbose = 0)
#
#y_test_predicted = model.predict(x_test, verbose = 0)
#
#print("--- Genetic Algorithm: %f minutes ---/n" % float((time.time() - start_time) / 60.0))
#
#











