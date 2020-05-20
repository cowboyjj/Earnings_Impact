

import pandas as pd
import numpy as np

import numpy as np
import FeatureSelection_NN as FS
import HyperparameterTuning_NN as HT
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
import time




#from keras.datasets import boston_housing
#(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
#mean = train_data.mean(axis=0)
#train_data -= mean
#std = train_data.std(axis=0)
#train_data /= std
#test_data -= mean
#test_data /= std
#x_train=train_data
#y_train=train_targets

start_time = time.time()


# Limit features to only those selected
#bestFeatures = [1 for i in range(featureCount)]
#x_train_FeatureSelected = []
#for point in x_train:
#    filtered = [i*j for i,j in zip(bestFeatures, point)]
#    x_train_FeatureSelected.append(np.array(filtered))
#x_train_FeatureSelected = np.array(x_train_FeatureSelected)
x_train_FeatureSelected = x_train
featureCount = len(x_train_FeatureSelected.columns)


#bestParams_All_1 = []
#generationIdx_All_1 = []
#currentAverageFitness_All_1 = []
#averageFitnessLog_All_1 = []
#allFitnessLog_All_1 = []
#
#for i in range(1):
#    
#    populationCount_HT = 20
#    generationCount_HT = 40
#    batchSize = 512                    
#
#    ret = HT.PerformHyperparameterTuning(x_train_FeatureSelected, 
#                                         y_train, 
#                                         featureCount, 
#                                         batchSize, 
#                                         populationCount_HT, 
#                                         generationCount_HT)
# 
#    averageFitnessLog_All_1.append(ret[3])
#    allFitnessLog_All_1.append(ret[4])
#    bestParams_All_1.append(ret[0] )
#    generationIdx_All_1.append(ret[1] )
#    currentAverageFitness_All_1.append(ret[2])
#    
    
    
    
#==============================================#
# Hyperparameter tuning
#==============================================#

populationCount_HT = 25
generationCount_HT = 6
batchSize = 512 
ACTIVATION = 'relu'
ret = HT.PerformHyperparameterTuning(x_train_FeatureSelected, y_train, batchSize, populationCount_HT, generationCount_HT)
tunedParameters = ret[0]

#==============================================#
# Test out-of-sample performance
#==============================================#

# Use the whole training set to re-train the model using the selected set of features and tuned hyperparameters
epochs_count = int(tunedParameters[0])
hiddenLayerNeuronCount = int(tunedParameters[1])
dropoutRate = tunedParameters[2]
reg_lambda = tunedParameters[3]
learningRate = tunedParameters[4]
layerCount = int(tunedParameters[5])
print("Tuned Params: epochsCount = %i, neuronCount = %i, dropout = %f, lambda = %f, learningRate = %f, layerCount = %i" % (epochs_count, hiddenLayerNeuronCount, dropoutRate, reg_lambda, learningRate, layerCount) )

# Layer 1
model = Sequential()
#model.add(Dense(hiddenLayerNeuronCount, kernel_regularizer=regularizers.l2(reg_lambda), input_dim = featureCount, activation='tanh'))
model.add(Dense(hiddenLayerNeuronCount, input_dim = featureCount, activation = ACTIVATION))
#model.add(Dropout(dropoutRate))

# Mid Layers
for i in range(layerCount):
    model.add(Dense(hiddenLayerNeuronCount, activation = ACTIVATION))
    #model.add(Dropout(dropoutRate))

# Output Layer
model.add(Dense(1, activation='linear'))
model.compile(loss='mae', optimizer='adam', metrics = ['mse']) # calculate additional metric 'mean absolute error'

# Standardize the training data
scaler = StandardScaler()
scaler.fit(x_train_FeatureSelected)
x_train_FeatureSelected_standardized = pd.DataFrame(data = scaler.transform(x_train_FeatureSelected),
                                                    index = x_train_FeatureSelected.index,
                                                    columns = x_train_FeatureSelected.columns)
                
# Also standardize the validation data, using the same scaler.      
x_test_standardized = pd.DataFrame(data = scaler.transform(x_test),
                                index = x_test.index,
                                columns = x_test.columns)

result_TestSet = model.fit(x_train_FeatureSelected_standardized, y_train, epochs = epochs_count, batch_size = batchSize, verbose = 0)
result_TestSet_history = result_TestSet.history

# Predict using the un-used test set data to which feature selection has been applied
#x_test_filtered_temp = []
#for point in x_test_standardized:
#    filtered = [i*j for i,j in zip(bestFeatures, point)]
#    x_test_filtered_temp.append(np.array(filtered))
#x_test_filtered = np.array(x_test_filtered_temp)
x_test_filtered = x_test_standardized
y_predicted = model.predict(x_test_filtered, verbose = 0)

print("\nUnique:")
print(len(np.unique(y_predicted)))

# Calculate result metrics
# https://scikit-learn.org/stable/modules/model_evaluation.html

from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_test, y_predicted)
rmse = mean_squared_error(y_test, y_predicted)
print("\nmae: %.3f%%, rmse: %.3f%% \n" % (100*mae, 100*rmse))
