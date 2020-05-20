

import pandas as pd
from copy import deepcopy
import numpy as np
import HyperparameterTuning_XG as HT
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import time
import xgboost as xgb
import random


start_time = time.time()


#==============Controls==============

useFeatureSelection = True
performModelTuning = False

#==============Controls==============


'''
1. +++++ Compute weights of imbalanced data - when performing Classification +++++
'''
sample_weight = np.ones(len(y_train))
if TestType != "Regression" or TestType == "Classification":
    classCount = len(np.unique(y_train))
    #class_weight = compute_class_weight('balanced', np.unique(y_train.T.values[0]) ,y_train.T.values[0])
    class_weight = compute_class_weight('balanced', np.unique(y_train.T.values) ,y_train.T.values)
    
    weightbase = class_weight[0]
    for index, w in np.ndenumerate(class_weight):
        class_weight[index] = w/weightbase
    
    for index, y in np.ndenumerate(y_train.T.values):    
        sample_weight[index] = class_weight[int(y)]
    
'''
2. +++++ Feature Selection +++++
'''
results = {}
bestFeatures = []
for i in range(500):
    
    x_train_FeatureSelected = []
    x_test_FeatureSelected = []
    if useFeatureSelection == False:
        x_train_FeatureSelected = x_train
        x_test_FeatureSelected = x_test
    else:    
                
        bestFeatures = np.random.randint(2, size = featureCount )
        
        
        # Limit features to only those selected
        columnsToDrop = []
        for idx in range(featureCount):
            val = bestFeatures[idx]
            if val == 0:
                columnsToDrop.append(idx)
                
        x_train_FeatureSelected = deepcopy(x_train)
        x_train_FeatureSelected = x_train_FeatureSelected.drop(x_train.columns[columnsToDrop], axis=1)
        
        x_test_FeatureSelected = deepcopy(x_test)
        x_test_FeatureSelected = x_test_FeatureSelected.drop(x_train.columns[columnsToDrop], axis=1)
        
    
    '''
    3. +++++ Start tuning +++++
    '''
    n_folds = 5
    early_stopping = 10
    n_estimators = 4000
    
    maxDepth_in = int()
    subSampleRate_in = float()
    colSampleRate_in = float()
    gamma_in = float()
    learningRate_in = float()
    min_child_weight_in = float()
    
    if performModelTuning == True:
        
        ######################################
        populationCount_HT = 2
        generationCount_HT = 1
        ######################################
        
        bestParams_All_1 = []
        generationIdx_All_1 = []
        minFitness_All_1 = []
        allFitnessLog_All_1 = []
    
        ret = HT.PerformHyperparameterTuning(TestType,
                                             x_train_FeatureSelected, 
                                             y_train, 
                                             sample_weight,
                                             n_estimators,
                                             n_folds,
                                             early_stopping,
                                             populationCount_HT, 
                                             generationCount_HT)
        
        bestParams_All_1.append(ret[0] )
        generationIdx_All_1.append(ret[1] )
        minFitness_All_1.append(ret[2])
        allFitnessLog_All_1.append(ret[3])
        
        # Extract the tuned parameters for later use
        bestParams = ret[0]
        
        maxDepth_in = int(bestParams[0])
        subSampleRate_in = bestParams[1]
        colSampleRate_in = bestParams[2]
        gamma_in = bestParams[3]
        learningRate_in = bestParams[4]
        min_child_weight_in = bestParams[5]
    else:
        maxDepth_in = 1
        subSampleRate_in = 0.8
        colSampleRate_in = 0.38
        gamma_in = 0.38
        learningRate_in = 0.7
        min_child_weight_in = 1
        
    print("\nGA results:eta = %f, max_depth = %i, colsample_bytree = %f, subsample = %f, gamma = %f, min_child_weight = %i" % (learningRate_in, maxDepth_in, colSampleRate_in, subSampleRate_in, gamma_in, min_child_weight_in) )    
    
         
        
    '''
    4. +++++ Train model using tuned params, then Predict +++++
    '''
    numRounds = 3
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
        
        xgb_reg.fit(x_train_FeatureSelected, 
                    y_train, 
                    eval_set=[(x_train_FeatureSelected, y_train)], 
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
            metrics = ['error']
        else:
            objective = 'multi:softprob'
            metrics = ['merror']
            
        # Complete params
        params = {'eta': learningRate_in, 
                  'max_depth': maxDepth_in, 
                  'colsample_bytree': colSampleRate_in, 
                  'subsample': subSampleRate_in,               
                  'gamma':gamma_in,              
                  'eval_metric': metrics,               
                  'silent': 1, 
                  'nthread':4,
                  'objective': objective}
        
        if classCount > 2:
            params['num_class'] = classCount
                
            
        '''
        b. ======== Apply tuned parameters to train model and perform prediction based on test dat ========
        '''
        
        traindata_matrix = xgb.DMatrix(x_train_FeatureSelected, label = y_train, weight = sample_weight)
    
        print("\nEvaluation stats during actual model training:")
        evals_result = {}
        watchlist = []
        watchlist.append((traindata_matrix, "TrainSet"))
        
        # early_stopping_rounds can be set in this method
        xg_trained = xgb.train(params, 
                               traindata_matrix, 
                               num_boost_round = numRounds, 
                               evals=watchlist, 
                               evals_result=evals_result)
    
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
    
        
        
        results[accuracy] = bestFeatures
        
        
        
    
        
              
    
    
    # Plot feature importance chart
#    xgb.plot_importance(xg_trained)
#    import matplotlib.pyplot as plt
#    plt.rcParams['figure.figsize'] = [5, 5]
#    plt.show()
    
    
    
    


