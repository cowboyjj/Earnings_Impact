

import pandas as pd
import numpy as np

import numpy as np
import FeatureSelection as FS
import HyperparameterTuning as HT
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
import time



start_time = time.time()
#data = trainingData["Post release OPEN - 1D"]
#featureNames = data["Feature Names"]
#inputPoints = data["Input Points"]
#outputPoints = data["Output Points"]
#featureCount = len(featureNames)



# Split the test data into training set and test set. Test set is not involved in the GA process.
x_train, x_test, y_train, y_test = train_test_split(np.array([np.array(x) for x in inputPoints]),
                                                    outputPoints,
                                                    test_size = 0.2)

bestFeatures_All_1 = []
generationIdx_All_1 = []
currentAverageFitness_All_1 = []
for i in range(6):
    # Feature selection
    populationCount_FS = len(featureNames)
    generationCount_FS = 20
    batchSize = 1028                    # 1028
    epochs_count = 5       
    hiddenLayerNeuronCount = 64         # 64
    numberOfMutatedGenes = 5
    ret = FS.PerformFeatureSelection(x_train, y_train, len(featureNames), batchSize, epochs_count, hiddenLayerNeuronCount, populationCount_FS, generationCount_FS, numberOfMutatedGenes)
    bestFeatures = ret[0]
    generationIdx = ret[1] 
    currentAverageFitness = ret[2]
    bestFeatures_All_1.append(bestFeatures )
    generationIdx_All_1.append(generationIdx)
    currentAverageFitness_All_1.append(currentAverageFitness)
    
#bestFeatures_All_2 = []
#generationIdx_All_2 = []
#currentAverageFitness_All_2 = []
#for i in range(6):
#    # Feature selection
#    populationCount_FS = len(featureNames)
#    generationCount_FS = 20
#    batchSize = 1028
#    epochs_count = 10    
#    hiddenLayerNeuronCount = 16
#    numberOfMutatedGenes = 5
#    ret = FS.PerformFeatureSelection(x_train, y_train, len(featureNames), batchSize, epochs_count, hiddenLayerNeuronCount, populationCount_FS, generationCount_FS, numberOfMutatedGenes)
#    bestFeatures = ret[0]
#    generationIdx = ret[1] 
#    currentAverageFitness = ret[2]
#    bestFeatures_All_2.append(bestFeatures )
#    generationIdx_All_2.append(generationIdx)
#    currentAverageFitness_All_2.append(currentAverageFitness)
#
#bestFeatures_All_3 = []
#generationIdx_All_3 = []
#currentAverageFitness_All_3 = []
#for i in range(6):
#    # Feature selection
#    populationCount_FS = len(featureNames)
#    generationCount_FS = 20
#    batchSize = 1028
#    epochs_count = 5
#    hiddenLayerNeuronCount = 32
#    numberOfMutatedGenes = 5
#    ret = FS.PerformFeatureSelection(x_train, y_train, len(featureNames), batchSize, epochs_count, hiddenLayerNeuronCount, populationCount_FS, generationCount_FS, numberOfMutatedGenes)
#    bestFeatures = ret[0]
#    generationIdx = ret[1] 
#    currentAverageFitness = ret[2]
#    bestFeatures_All_3.append(bestFeatures )
#    generationIdx_All_3.append(generationIdx)
#    currentAverageFitness_All_3.append(currentAverageFitness)
#
#bestFeatures_All_4 = []
#generationIdx_All_4 = []
#currentAverageFitness_All_4 = []
#for i in range(6):
#    # Feature selection
#    populationCount_FS = len(featureNames)
#    generationCount_FS = 20
#    batchSize = 1028
#    epochs_count = 5
#    hiddenLayerNeuronCount = 16
#    numberOfMutatedGenes = 5
#    ret = FS.PerformFeatureSelection(x_train, y_train, len(featureNames), batchSize, epochs_count, hiddenLayerNeuronCount, populationCount_FS, generationCount_FS, numberOfMutatedGenes)
#    bestFeatures = ret[0]
#    generationIdx = ret[1] 
#    currentAverageFitness = ret[2]
#    bestFeatures_All_4.append(bestFeatures )
#    generationIdx_All_4.append(generationIdx)
#    currentAverageFitness_All_4.append(currentAverageFitness)
#    
#
#bestFeatures_All_5 = []
#generationIdx_All_5 = []
#currentAverageFitness_All_5 = []
#for i in range(6):
#    # Feature selection
#    populationCount_FS = len(featureNames)
#    generationCount_FS = 20
#    batchSize = 1028
#    epochs_count = 5
#    hiddenLayerNeuronCount = 64
#    numberOfMutatedGenes = 5
#    ret = FS.PerformFeatureSelection(x_train, y_train, len(featureNames), batchSize, epochs_count, hiddenLayerNeuronCount, populationCount_FS, generationCount_FS, numberOfMutatedGenes)
#    bestFeatures = ret[0]
#    generationIdx = ret[1] 
#    currentAverageFitness = ret[2]
#    bestFeatures_All_5.append(bestFeatures )
#    generationIdx_All_5.append(generationIdx)
#    currentAverageFitness_All_5.append(currentAverageFitness)
#
#bestFeatures_All_6 = []
#generationIdx_All_6 = []
#currentAverageFitness_All_6 = []
#for i in range(6):
#    # Feature selection
#    populationCount_FS = len(featureNames)
#    generationCount_FS = 20
#    batchSize = 1028
#    epochs_count = 10
#    hiddenLayerNeuronCount = 64
#    numberOfMutatedGenes = 5
#    ret = FS.PerformFeatureSelection(x_train, y_train, len(featureNames), batchSize, epochs_count, hiddenLayerNeuronCount, populationCount_FS, generationCount_FS, numberOfMutatedGenes)
#    bestFeatures = ret[0]
#    generationIdx = ret[1] 
#    currentAverageFitness = ret[2]
#    bestFeatures_All_6.append(bestFeatures )
#    generationIdx_All_6.append(generationIdx)
#    currentAverageFitness_All_6.append(currentAverageFitness)
#    
#    
###########
#    
#
#bestFeatures_All_7 = []
#generationIdx_All_7 = []
#currentAverageFitness_All_7 = []
#for i in range(6):
#    # Feature selection
#    populationCount_FS = len(featureNames)
#    generationCount_FS = 20
#    batchSize = 1028
#    epochs_count = 10    
#    hiddenLayerNeuronCount = 32
#    numberOfMutatedGenes = 5
#    ret = FS.PerformFeatureSelection(x_train, y_train, len(featureNames), batchSize, epochs_count, hiddenLayerNeuronCount, populationCount_FS, generationCount_FS, numberOfMutatedGenes, False)
#    bestFeatures = ret[0]
#    generationIdx = ret[1] 
#    currentAverageFitness = ret[2]
#    bestFeatures_All_7.append(bestFeatures )
#    generationIdx_All_7.append(generationIdx)
#    currentAverageFitness_All_7.append(currentAverageFitness)
#    
#bestFeatures_All_8 = []
#generationIdx_All_8 = []
#currentAverageFitness_All_8 = []
#for i in range(6):
#    # Feature selection
#    populationCount_FS = len(featureNames)
#    generationCount_FS = 20
#    batchSize = 1028
#    epochs_count = 10    
#    hiddenLayerNeuronCount = 16
#    numberOfMutatedGenes = 5
#    ret = FS.PerformFeatureSelection(x_train, y_train, len(featureNames), batchSize, epochs_count, hiddenLayerNeuronCount, populationCount_FS, generationCount_FS, numberOfMutatedGenes, False)
#    bestFeatures = ret[0]
#    generationIdx = ret[1] 
#    currentAverageFitness = ret[2]
#    bestFeatures_All_8.append(bestFeatures )
#    generationIdx_All_8.append(generationIdx)
#    currentAverageFitness_All_8.append(currentAverageFitness)
#
#bestFeatures_All_9 = []
#generationIdx_All_9 = []
#currentAverageFitness_All_9 = []
#for i in range(6):
#    # Feature selection
#    populationCount_FS = len(featureNames)
#    generationCount_FS = 20
#    batchSize = 1028
#    epochs_count = 5
#    hiddenLayerNeuronCount = 32
#    numberOfMutatedGenes = 5
#    ret = FS.PerformFeatureSelection(x_train, y_train, len(featureNames), batchSize, epochs_count, hiddenLayerNeuronCount, populationCount_FS, generationCount_FS, numberOfMutatedGenes, False)
#    bestFeatures = ret[0]
#    generationIdx = ret[1] 
#    currentAverageFitness = ret[2]
#    bestFeatures_All_9.append(bestFeatures )
#    generationIdx_All_9.append(generationIdx)
#    currentAverageFitness_All_9.append(currentAverageFitness)
#
#bestFeatures_All_10 = []
#generationIdx_All_10 = []
#currentAverageFitness_All_10 = []
#for i in range(6):
#    # Feature selection
#    populationCount_FS = len(featureNames)
#    generationCount_FS = 20
#    batchSize = 1028
#    epochs_count = 5
#    hiddenLayerNeuronCount = 16
#    numberOfMutatedGenes = 5
#    ret = FS.PerformFeatureSelection(x_train, y_train, len(featureNames), batchSize, epochs_count, hiddenLayerNeuronCount, populationCount_FS, generationCount_FS, numberOfMutatedGenes, False)
#    bestFeatures = ret[0]
#    generationIdx = ret[1] 
#    currentAverageFitness = ret[2]
#    bestFeatures_All_10.append(bestFeatures )
#    generationIdx_All_10.append(generationIdx)
#    currentAverageFitness_All_10.append(currentAverageFitness)
#
#
#bestFeatures_All_11 = []
#generationIdx_All_11 = []
#currentAverageFitness_All_11 = []
#for i in range(6):
#    # Feature selection
#    populationCount_FS = len(featureNames)
#    generationCount_FS = 20
#    batchSize = 1028
#    epochs_count = 5
#    hiddenLayerNeuronCount = 64
#    numberOfMutatedGenes = 5
#    ret = FS.PerformFeatureSelection(x_train, y_train, len(featureNames), batchSize, epochs_count, hiddenLayerNeuronCount, populationCount_FS, generationCount_FS, numberOfMutatedGenes, False)
#    bestFeatures = ret[0]
#    generationIdx = ret[1] 
#    currentAverageFitness = ret[2]
#    bestFeatures_All_11.append(bestFeatures )
#    generationIdx_All_11.append(generationIdx)
#    currentAverageFitness_All_11.append(currentAverageFitness)
#
#bestFeatures_All_12 = []
#generationIdx_All_12 = []
#currentAverageFitness_All_12 = []
#for i in range(6):
#    print ("i = %i" % i)
#    # Feature selection
#    populationCount_FS = len(featureNames)
#    generationCount_FS = 20
#    batchSize = 1028
#    epochs_count = 10
#    hiddenLayerNeuronCount = 64
#    numberOfMutatedGenes = 5
#    ret = FS.PerformFeatureSelection(x_train, y_train, len(featureNames), batchSize, epochs_count, hiddenLayerNeuronCount, populationCount_FS, generationCount_FS, numberOfMutatedGenes, False)
#    bestFeatures = ret[0]
#    generationIdx = ret[1] 
#    currentAverageFitness = ret[2]
#    bestFeatures_All_12.append(bestFeatures )
#    generationIdx_All_12.append(generationIdx)
#    currentAverageFitness_All_12.append(currentAverageFitness)
#    
    
#######
    
