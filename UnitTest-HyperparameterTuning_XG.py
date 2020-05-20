
# ** Parameter description ** 
# https://xgboost.readthedocs.io/en/latest/parameter.html
# https://xgboost.readthedocs.io/en/latest/python/python_api.html
# https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters
# https://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf (technical doc)

# ** Good examples **
# **** https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f ****
# https://www.kaggle.com/tilii7/bias-correction-xgboost
# https://github.com/dmlc/xgboost/blob/master/demo/guide-python/cross_validation.py
# https://datascience.stackexchange.com/questions/16342/unbalanced-multiclass-data-with-xgboost   (about weighting imbalanced data)


import pandas as pd
from copy import deepcopy
import numpy as np
import HyperparameterTuning_XG as HT
from sklearn.preprocessing import RobustScaler, StandardScaler
from Utilities import computeSampleWeights
import time
import xgboost as xgb
import random


start_time = time.time()





#==============Controls==============
#==============Controls==============

inProcessFeatureSelection = False

performModelTuning = True

doSampleWeight = True

classificationMetric = 'error'

validationSetPercentage = 0.1

#
populationCount_HT = 15
generationCount_HT = 3
#

priorFeatureSelection = False
remaining_columns = ["priceReaction_minus1D_to_1D"]
#                        "EPS Earnings_Surprise_Backward_Ave_Diff", 
#                     "EPS Earnings_Surprise_Backward_Diff", 
#                     "EPS EarningsSurprise"]
#                     "Total Liabilities_Q_Change", 
#                     "Operating Income_Y_Change", 
#                     "Return On Common Equity", 
#                     "Gross Profit_Y_Change"]

remaining_columns = [i for i in x_train.columns if i not in remaining_columns]

#==============Controls==============
#==============Controls==============


    
'''
1. +++++ Feature Selection +++++
'''

x_train_FeatureSelected = []
x_test_FeatureSelected = []
if priorFeatureSelection == False:
    x_train_FeatureSelected = deepcopy(x_train)
    x_test_FeatureSelected = deepcopy(x_test)
else:    
    
    ######################### DEBUG #########################
    
#    x = 1
#    print("x = %i" % x)
#    a = [1] * x
#    b = [0] * (featureCount -x)
#    
#    bestFeatures = []
#    bestFeatures = a + b
#    random.shuffle(bestFeatures)
    
    
#x = random.randint(1,featureCount)    
#    featureCount = len(x_train.columns)
#    bestFeatures = np.random.randint(2, size = featureCount )
    
#    # Limit features to only those selected
#    columnsToDrop = []
#    for idx in range(featureCount):
#        val = bestFeatures[idx]
#        if val == 0:
#            columnsToDrop.append(idx)
#            
#    x_train_FeatureSelected = deepcopy(x_train)
#    x_train_FeatureSelected = x_train_FeatureSelected.drop(x_train.columns[columnsToDrop], axis=1)
#    
#    x_test_FeatureSelected = deepcopy(x_test)
#    x_test_FeatureSelected = x_test_FeatureSelected.drop(x_train.columns[columnsToDrop], axis=1)
    
    ######################### DEBUG #########################
           
    x_train_FeatureSelected = deepcopy(x_train)
    x_test_FeatureSelected = deepcopy(x_test)
    
    x_train_FeatureSelected = x_train.filter(remaining_columns)
    x_test_FeatureSelected = x_test.filter(remaining_columns)
    

'''
2. +++++ Separate train, validation, and test data +++++
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

#x_train_val = x_train_FeatureSelected
#y_train_val = y_train
#
#x_train_final = x_train_FeatureSelected
#y_train_final = y_train


'''
3. +++++ Compute weights of imbalanced data - when performing Classification +++++
'''
classCount = len(np.unique(y_train))
sample_weight = computeSampleWeights(y_train)
sample_weight_val = sample_weight[:validSetSize]
sample_weight_training = sample_weight[(validSetSize+1):]


'''
4. +++++ Start tuning +++++
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
    
#else:
#    maxDepth_in = 1
#    subSampleRate_in = 0.8
#    colSampleRate_in = 0.38
#    gamma_in = 0.38
#    learningRate_in = 0.7
#    min_child_weight_in = 1
    
print("\nGA results:eta = %f, max_depth = %i, colsample_bytree = %f, subsample = %f, gamma = %f, min_child_weight = %i" % (learningRate_in, maxDepth_in, colSampleRate_in, subSampleRate_in, gamma_in, min_child_weight_in) )    

     
    
'''
4. +++++ Train model using tuned params, then Predict +++++
'''

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
    
    # Calculate result metrics
    # https://scikit-learn.org/stable/modules/model_evaluation.html
    
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
    
#    traindata_matrix = xgb.DMatrix(x_train_FeatureSelected, label = y_train, weight = sample_weight)
#
#    print("\nEvaluation stats during actual model training:")
#    evals_result = {}
#    watchlist = []
#    watchlist.append((traindata_matrix, "TrainSet"))
#    
#    # early_stopping_rounds can be set in this method
#    xg_trained = xgb.train(params, 
#                           traindata_matrix, 
#                           early_stopping_rounds = early_stopping,
#                           num_boost_round = 35,    #n_estimators, 
#                           evals=watchlist, 
#                           evals_result=evals_result)
    
    
    
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
            
    accuracy = accuracy_score(y_test, np.array(y_predicted_MultiClass))            
    print("\nAccuracy: %.2f%%\n" % (accuracy * 100.0))

    precision = precision_score(y_test, y_predicted_MultiClass,average='micro')     # true positive / (true positive + false positive)
    recall = recall_score(y_test, y_predicted_MultiClass, average='micro')          # true positive / (true positive + false negative)
    print("--precision = %f, recall = %f\n" % (precision, recall))
    
    if classCount == 2:
        precision = precision_score(y_test, y_predicted_MultiClass,average='micro')     # true positive / (true positive + false positive)
        recall = recall_score(y_test, y_predicted_MultiClass, average='micro')          # true positive / (true positive + false negative)
        print("--precision = %f, recall = %f\n" % (precision, recall))

    target_names = ['class ' + str(i) for i in np.unique(y_train)]
    print(classification_report(y_test, y_predicted_MultiClass, target_names=target_names))
    
    
    
# Plot feature importance
# Examples: https://www.programcreek.com/python/example/99827/xgboost.plot_importance
from xgboost import plot_importance
a = plot_importance(xg_trained, importance_type = 'gain', max_num_features=20) # top 10 most important features



#    # Plot feature graph    
#    import os
#    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
#    xgb.to_graphviz(xg_trained)
    
    
              
    
    
    # Plot feature importance chart
#    xgb.plot_importance(xg_trained)
#    import matplotlib.pyplot as plt
#    plt.rcParams['figure.figsize'] = [5, 5]
#    plt.show()
    
    
    
    


