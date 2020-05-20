
import pandas as pd

import numpy as np
import HyperparameterTuning_SVR as SVR_GA
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import time


start_time = time.time()


# Compute weights of imbalanced data - when performing Classification
sample_weight = np.ones(len(y_train))
if TestType != "Regression" or TestType == "Classification":
    classCount = len(np.unique(y_train))
    class_weight = compute_class_weight('balanced', np.unique(y_train.T.values[0]) ,y_train.T.values[0])
    
    weightbase = class_weight[0]
    for index, w in np.ndenumerate(class_weight):
        class_weight[index] = w/weightbaseCalais, France
    
    for index, y in np.ndenumerate(y_train.T.values[0]):    
        sample_weight[index] = class_weight[int(y)]
    
    
# Limit features to only those selected
#x_train_FeatureSelected = []
#for point in x_train:
#    filtered = []
#    for idx in range(len(bestFeatures)):
#        if point[i] == 'nan' or bestFeatures[i] == 0:
#            filtered.append('nan')
#        else:
#            filtered.append(point[i])
#    x_train_FeatureSelected.append(np.array(filtered))
#
#x_train_FeatureSelected = np.array(x_train_FeatureSelected)
            
x_train_FeatureSelected = x_train
#y_train = np.array(y_train)


    
'''
1. Use GA to pick the best performing hyperparameters
'''
populationCount_HT = 20
generationCount_HT = 4


'''
======== Start tuning ========
'''
ret = SVR_GA.PerformHyperparameterTuning(x_train_FeatureSelected, y_train, populationCount_HT, generationCount_HT)

# Extract the tuned parameters for later use
bestParams = ret[0]

kernel = str(bestParams[0])
gamma = float(bestParams[1])
C = float(bestParams[2])
epsilon = float(bestParams[3])

print("\nGA results: kernel = %s, gamma = %f, C = %f, epsilon = %f" % (kernel, gamma, C, epsilon) )
   
degree = 3 
if kernel == 'poly2':
    kernel = 'poly'
    degree = 2
elif kernel == 'poly3':
    kernel = 'poly'
    degree = 3
elif kernel == 'poly4':
    kernel = 'poly'
    degree = 4
elif kernel == 'poly5':
    kernel = 'poly'
    degree = 5
    
kernel = 'linear'
gamma = 0.01
C = 0.0206913808111479
epsilon = 0.05

# Train model using tuned params
if TestType == "Regression":
        
    scaler = StandardScaler()
    scaler.fit(x_train_FeatureSelected)
    
    # Standardize the training data
    x_train_FeatureSelected_standardized = pd.DataFrame(data = scaler.transform(x_train_FeatureSelected),
                                                        index = x_train_FeatureSelected.index,
                                                        columns = x_train_FeatureSelected.columns)
                    
    # Also standardize the validation data, using the same scaler.      
    x_test_standardized = pd.DataFrame(data = scaler.transform(x_test),
                                    index = x_test.index,
                                    columns = x_test.columns)
    
    model = SVR(kernel = kernel, gamma = gamma, C = C, epsilon = epsilon, degree = degree, verbose=True, max_iter = 1000, cache_size = 3000)
    model.fit(x_train_FeatureSelected_standardized, y_train)
    
    y_predicted = model.predict(x_test_standardized)
        
    print("\nUnique:")
    print(len(np.unique(y_predicted)))
    
    # Calculate result metrics
    # https://scikit-learn.org/stable/modules/model_evaluation.html
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(y_test, y_predicted)
    rmse = mean_squared_error(y_test, y_predicted)
    print("\nmae: %.3f%%, rmse: %.3f%% \n" % (100*mae, 100*rmse))
    
else:
    
    ###
    i = 1
    


