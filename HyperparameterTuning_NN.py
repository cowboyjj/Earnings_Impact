import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import backend as K
from keras import regularizers
from keras import optimizers
from copy import deepcopy

import random
import time



class GA_HyperparameterTuning:
    
    def __init__(self, fold_count, number_of_mutated_genes, batchSize, featureCount, mutationRate):
        self.fold_count = fold_count
        self.batchSize = batchSize
        self.number_of_mutated_genes = number_of_mutated_genes
        self.mutationRate = mutationRate          
        self.featureCount = featureCount
        
        # All model hyperparameters are initialised as 0
        self.epochs_count = 0 
        self.hiddenLayerNeuronCount = 0 
        self.dropoutRate = 0 
        self.reg_lambda = 0
        self.learningRate = 0
        self.layerCount = 0
        
    def initModel(self):
        # define the final model        
        
        self.model = Sequential()     
        ACTIVATION = 'relu'
        
        # Layer 1        
        self.model.add(Dense(self.hiddenLayerNeuronCount, input_dim = self.featureCount, activation = ACTIVATION))
#        self.model.add(Dense(self.hiddenLayerNeuronCount, kernel_regularizer=regularizers.l2(self.reg_lambda), activation='tanh'))
#        self.model.add(Dropout(self.dropoutRate))
        
        # Mid Layers
        for i in range(self.layerCount):
            self.model.add(Dense(self.hiddenLayerNeuronCount, activation = ACTIVATION))        ############### NO Regularization ############### 
#            self.model.add(Dropout(self.dropoutRate))

        # Output Layer 
        self.model.add(Dense(1, activation='linear'))
        
#        self.model.compile(loss='mse', optimizer='adam', metrics = ['mae']) # calculate additional metric 'mean absolute error'
        adam = optimizers.Adam(lr = self.learningRate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) #0.001
        self.model.compile(loss='mae', optimizer = adam, metrics = ['mse']) # calculate additional metric 'mean absolute error'
     

    
    def calcPopulationFitness(self, x_train, y_train, new_population):

        folds = KFold(n_splits = self.fold_count, shuffle = True)#, random_state = 100)
        
        fitness = []
        lossValuePerPopulation = []
        for i in range(new_population.shape[0]):
            chromosome = new_population[i]            
            
            # Initialise hyperparameters according to settings in the current chromosome 
            self.epochs_count = int(chromosome[0])
            self.hiddenLayerNeuronCount = int(chromosome[1])
            self.dropoutRate = chromosome[2]
            self.reg_lambda = chromosome[3]
            self.learningRate = chromosome[4]
            self.layerCount = int(chromosome[5])
                        
            # Calculate the k-fold validation loss as the fitness value
            val_loss = 0.0
            loss = 0.0
            mae = 0.0
                
            print("--Chromosome: %i, epochsCount = %i, neuronCount = %i, dropout = %f, lambda = %f, learningRate = %f, layerCount = %i" % (i, self.epochs_count, self.hiddenLayerNeuronCount, self.dropoutRate, self.reg_lambda, self.learningRate, self.layerCount) )

            idx = 0
            for train_index, validate_index in folds.split(x_train):
                   
                start_time = time.time()            
                idx = idx + 1
                
                x_train_sub = x_train.iloc[train_index, :]
                y_train_sub = y_train.iloc[train_index]            
                
                x_validate  = x_train.iloc[validate_index, :]
                y_validate  = y_train.iloc[validate_index]
                
#                x_train_sub = np.array(x_train)[train_index]
#                y_train_sub = np.array(y_train)[train_index]            
#                
#                x_validate  = np.array(x_train)[validate_index]
#                y_validate  = np.array(y_train)[validate_index]
                
            
                # For each fold, standardize the training data.           
                scaler = StandardScaler()
                scaler.fit(x_train_sub)
                
                x_train_sub = pd.DataFrame(data = scaler.transform(x_train_sub),
                                                index = x_train_sub.index,
                                                columns = x_train_sub.columns)
                                
                # Also standardize the validation data, using the same scaler.      
                x_validate = pd.DataFrame(data = scaler.transform(x_validate),
                                                index = x_validate.index,
                                                columns = x_validate.columns)
                
                # Initialize learning model
                self.initModel()
            
                # fit the final model
                fitResult = self.model.fit(x_train_sub, 
                                           y_train_sub, 
                                           epochs = self.epochs_count, 
                                           batch_size = self.batchSize, 
                                           verbose = 0, 
                                           validation_data=(x_validate, y_validate))
            
                val_loss_current = fitResult.history["val_loss"][-1]
                loss_current = fitResult.history["loss"][-1]
                #mae_current = fitResult.history["mean_absolute_error"][-1]
                
                val_loss = val_loss + val_loss_current
                loss = loss + loss_current
                #mae = mae + mae_current
                
                #print("----Fold %s, took %f seconds, val_loss = %f, loss = %f, mae = %f" % (idx, time.time() - start_time, val_loss_current, loss_current, mae_current))
                print("----Fold %s, took %f seconds, val_loss = %f, loss = %f" % (idx, time.time() - start_time, val_loss_current, loss_current))
        
            val_loss = val_loss/self.fold_count        
            loss = loss / self.fold_count
            #mae = mae / self.fold_count
            #print("--Chromosome: %i, ave(val_loss) = %f, ave(loss) = %f, ave(mae) = %f" % (i, val_loss, loss, mae))
            print("--Chromosome: %i, mae_val = %f, mae = %f" % (i, val_loss, loss))
            print("\n")
            
            # Use Validation Loss value as the fitness value
            fitness.append(val_loss)
            lossValuePerPopulation.append(loss)
            
            # Clear all the models created so far otherwise TF will get run slower
            K.clear_session()
            
        return (fitness, lossValuePerPopulation)
    
        
    def select_mating_pool(self, pop, fitness, num_parents):
        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        parents = np.empty((num_parents, pop.shape[1]))
        for parent_num in range(num_parents):
            # Within each loop find the chromosome within the current generation with the smallest fitness
            # and assign this chromosome to the 'parents' list. This repeats for 'num_parents' times.
            min_fitness_idx = np.where(fitness == np.min(fitness))
            min_fitness_idx = min_fitness_idx[0][0]
            parents[parent_num, :] = pop[min_fitness_idx, :]
            fitness[min_fitness_idx] = 99999999999
        return parents
    
    
    def crossover(self, parents, offspring_size):
        offspring = np.empty(offspring_size)
        # The point at which crossover takes place between two parents. Usually it is at the center.
        crossover_point = np.uint8(offspring_size[1]/2)
    
        for k in range(offspring_size[0]):
            # Index of the first parent to mate.
            parent1_idx = k%parents.shape[0]
            # Index of the second parent to mate.
            parent2_idx = (k+1)%parents.shape[0]
            # The new offspring will have its first half of its genes taken from the first parent.
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            # The new offspring will have its second half of its genes taken from the second parent.
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        return offspring
    
    
    def mutation(self, offspring_crossover):
        
        geneCount = offspring_crossover.shape[1]
        
        for idx in range(offspring_crossover.shape[0]):
            if random.random() < self.mutationRate:
                
                print("*********** Mutating ***********")
                chromosome = offspring_crossover[idx]
                for i in range(self.number_of_mutated_genes):
                    mutateIdx = random.randint(0, geneCount-1)
                    chromosome[mutateIdx] = self.mutateGene(mutateIdx)
                
                offspring_crossover[idx] = chromosome
    
        return offspring_crossover         
       
       
    def initializePopulation(self, populationCount):
        population = []
        for i in range(populationCount):
            newChromosome = self.generateNewChromosome()
            population.append(np.array(newChromosome))
        
        return np.array(population)
    
    
    def generateNewChromosome(self):
        epochs_count           = int(self.mutateGene(0))
        hiddenLayerNeuronCount = int(self.mutateGene(1))
        dropoutRate             = self.mutateGene(2)
        reg_lambda              = self.mutateGene(3)
        learningRate            = self.mutateGene(4)
        layerCount              = int(self.mutateGene(5))
        
        return [epochs_count, hiddenLayerNeuronCount, dropoutRate, reg_lambda, learningRate, layerCount]
    
    
    def mutateGene(self, geneIdx):
        
        if geneIdx == 0:    # epochs_count
            newGene = int
            newGene = random.randint(1,100)
            
        elif geneIdx == 1:  # hidden layer neuron count
            newGene = int
            newGene = random.randint(3,60)
            return newGene
        
        elif geneIdx == 2:  # droput rate
            newGene = float
            newGene = random.randint(0,20) * 0.05
            return newGene
        
        elif geneIdx == 3:  # regularization lambda
            newGene = float
            newGene = 0 #random.randint(0,20) * 0.0005
        
        elif geneIdx == 4:  # learning rate
            newGene = float
            newGene = random.randint(0,300) * 0.005
            
        elif geneIdx == 5:  # mid layer count
            newGene = random.randint(0,12)
            
        return newGene
    
    
def PerformHyperparameterTuning(x_train_featureSelected, 
                                y_train_featureSelected, 
                                batch_size, 
                                population_count, 
                                generation_count):
    
    populationCount = population_count
    generationCount = generation_count
    fold_count = 5
    batchSize = batch_size
    number_of_mutated_genes = 4
    mutationRate = 0.3
    featureCount = len(x_train_featureSelected.columns)
        
    # Initialise GA object
    tuningGA = GA_HyperparameterTuning(fold_count, number_of_mutated_genes, batchSize, featureCount, mutationRate)
    population = tuningGA.initializePopulation(populationCount)
        
    # Loop through a GA routine to select the best combination of features
    num_parents_mating = int(populationCount / 2)
    fitness_allGenerations = []             # for DEBUG
    lossValue_allGenerations = []           # for DEBUG
    population_allGenerations = []          # for DEBUG
    
    minFitness = []
    lastAverageFitness = float('inf')
    averageFitnessLog = []
    
    for generationIdx in range(generationCount):
        
        print("Generation %s" % generationIdx)
        # 1. Measuring the fitness of each chromosome in the population.
        ret = tuningGA.calcPopulationFitness(x_train_featureSelected, y_train_featureSelected, population)
        
        fitness = ret[0]
        loss = ret[1]
        temp = deepcopy(fitness)
        fitness_allGenerations.append(temp)
        lossValue_allGenerations.append(loss)
        population_allGenerations.append(population)
        
        # If average fitness no longer changes, break off the routine
        minFitness.append(float(np.min(fitness)))   
        currentAverageFitness = np.array(minFitness).mean()
        averageFitnessLog.append(currentAverageFitness)
        print("--Generation %s. lastAverageFitness    = %f" % (generationIdx, lastAverageFitness) )
        print("--Generation %s, currentAverageFitness = %f \n" % (generationIdx, currentAverageFitness) )
        print("--Generation %s, minFitness = %f \n" % (generationIdx, float(np.min(fitness))) )
#        if abs((lastAverageFitness - currentAverageFitness)) < 0.00001:
#            print("========= converged =========")
#            print("========= converged =========")
#            print("========= converged =========")
#            break
        lastAverageFitness = currentAverageFitness
        
        # 2. Selecting the best parents in the population for mating.
        parents = tuningGA.select_mating_pool(population, fitness, num_parents_mating)
    
        # 3. Generating next generation using crossover.
        offspring_crossover = tuningGA.crossover(parents,
                                           offspring_size=(populationCount - parents.shape[0], parents.shape[1]))
    
        # 4. Adding some variations to the offsrping using mutation.
        offspring_mutation = tuningGA.mutation(offspring_crossover)
        
        # Creating the new population based on the parents and offspring.
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = offspring_mutation
    
    # Calculate fitness of the final set of chromosomes
    fitness = tuningGA.calcPopulationFitness(x_train_featureSelected, y_train_featureSelected, population)
    
    min_fitness_idx = np.where(fitness == np.min(fitness))
    min_fitness_idx = min_fitness_idx[0][0]
    tunedParameters = population[min_fitness_idx, :]
    return (tunedParameters, generationIdx, currentAverageFitness, averageFitnessLog, fitness_allGenerations)
    
    