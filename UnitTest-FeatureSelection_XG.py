

import pandas as pd
import numpy as np

import numpy as np
import FeatureSelection_XG as FS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import time
import random


start_time = time.time()


# Compute weights of imbalanced data - when performing Classification
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
        
        
bestFeatures = []
bestFeatures_All_1 = []
generationIdx_All_1 = []
minFitness_All_1 = []
allFitnessLog_All_1 = []
finalPopulation = []

for i in range(1):
    # Feature selection
    populationCount_HT = 20
    generationCount_HT = 20

    # Parameter description: https://xgboost.readthedocs.io/en/latest/parameter.html
    
    ret = FS.PerformFeatureSelection(TestType,
                                     x_train, 
                                     y_train, 
                                     sample_weight,
                                     populationCount_HT, 
                                     generationCount_HT)
 
    bestFeatures = ret[0]
    bestFeatures_All_1.append(ret[0] )
    generationIdx_All_1.append(ret[1] )
    minFitness_All_1.append(ret[2])
    allFitnessLog_All_1.append(ret[3])
    finalPopulation = ret[4]