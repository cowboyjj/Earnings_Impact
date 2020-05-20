import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler, StandardScaler
import xgboost as xgb
from copy import deepcopy
import random
import time
from scipy import stats


class GA_HyperparameterTuning:
    
    def __init__(self, testType, fold_count, number_of_mutated_genes, mutationRate, n_estimators, weight, earlyStoppingRound, featureCount, classificationMetric):
        self.fold_count = fold_count
        self.number_of_mutated_genes = number_of_mutated_genes
        self.mutationRate = mutationRate          
        self.testType = testType
        self.n_estimators = n_estimators
        self.earlyStoppingRound = earlyStoppingRound
        self.featureCount = featureCount
        self.max_depth = 0
        self.colsample_bytree = 0
        self.subsample = 0
        self.gamma = 0
        
        self.sample_weight = weight
        self.classificationMetric = classificationMetric

    
    def calcPopulationFitness(self, x_train, y_train, new_population, doFeatureSelection):
        
        # Define objective and metric types
        objective = ""
        metrics = []
        if self.testType == "Regression":
            objective = "reg:linear"  # linear regression
            #metrics = ['rmse', 'mae']
            metrics = ['mae']
        else:
            classCount = len(np.unique(y_train))
            if classCount == 2:
                objective = 'binary:logistic'
                #metrics = ['aucpr', 'map', 'error']
                metrics = [self.classificationMetric]
            else:
                objective = 'multi:softprob'
                metrics = ['merror']
            
        fitness = []
        for i in range(new_population.shape[0]):
            
            chromosome = new_population[i]
            
            #####################################################################
            # 1. Initialise hyperparameters according to settings in the current chromosome 
            #####################################################################
            self.max_depth = int(chromosome[0])
            self.subsample = float(chromosome[1])
            self.colsample_bytree = float(chromosome[2])
            self.gamma = float(chromosome[3])
            self.learningRate = float(chromosome[4])
            self.min_child_weight = int(chromosome[5])
            
            # Feature Selection
            x_train_FeatureSelected = deepcopy(x_train)
            if doFeatureSelection == True:
                # 1
                selectedFeatures = chromosome[6]    
                # 2
                columnsToDrop = []
                for idx in range(self.featureCount):
                    val = selectedFeatures[idx]
                    if val == 0:
                        columnsToDrop.append(idx)
                # 3
                x_train_FeatureSelected = x_train_FeatureSelected.drop(x_train.columns[columnsToDrop], axis=1)
                
            # Create training data DMatrix object   
            if self.testType == "Regression":
                xg_train = xgb.DMatrix(x_train_FeatureSelected, label=y_train)
            else:
                xg_train = xgb.DMatrix(x_train_FeatureSelected, label=y_train, weight = self.sample_weight)
            
            
            #####################################################################
            # 2. Set the parameters to XGBoost            
            #####################################################################
            print("--Chromosome: %i, eta = %f, max_depth = %i, colsample_bytree = %f, subsample = %f, gamma = %f, min_child_weight = %i" % (i, self.learningRate, self.max_depth, self.colsample_bytree, self.subsample, self.gamma, self.min_child_weight) )

            error = 0.0
            val_error = 0.0
            params = {'eta': self.learningRate, 
                      'max_depth': self.max_depth, 
                      'subsample': self.subsample, 
                      'colsample_bytree': self.colsample_bytree,
                      'gamma': self.gamma,
                      'eval_metric': metrics, 
                      'min_child_weight': self.min_child_weight,
                      'silent':1,
                      'nthread':4,
                      #'scale_pos_weight':1,
                      'objective': objective}            
            
            if self.testType != "Regression" and classCount > 2:
                params['num_class'] = classCount
                
            # 3. Cross validation
            CV = xgb.cv(params, 
                        xg_train, 
                        num_boost_round = self.n_estimators, 
                        nfold = self.fold_count, 
                        verbose_eval = False,
                        early_stopping_rounds = self.earlyStoppingRound)
            
            # 4. Estimate errors
            if self.testType == "Regression":
                val_error = list(CV['test-mae-mean'])[-1]
                error = list(CV['train-mae-mean'])[-1]
            else:
                if classCount == 2:
                    name = 'test-' + self.classificationMetric + '-mean'
                    val_error = list(CV[name])[-1]
                    error = list(CV[name])[-1]
                    
                    if self.classificationMetric == 'auc' or self.classificationMetric == 'aucpr':
                        val_error = -1 * val_error
                        error = -1 * error
                else:
                    val_error = list(CV['test-merror-mean'])[-1]
                    error = list(CV['train-merror-mean'])[-1]
                
            print("--Chromosome: %i, val_error = %f, error = %f" % (i, val_error, error))
            print("\n")
                        
            # Use Validation Loss value as the fitness value
            fitness.append(val_error)
#            fitness.append(error)
            
        return fitness
        
    def select_mating_pool(self, pop, fitness, num_parents):
        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        parents = np.empty((num_parents, pop.shape[1]), dtype=object)
        for parent_num in range(num_parents):
            # Within each loop find the chromosome within the current generation with the smallest fitness
            # and assign this chromosome to the 'parents' list. This repeats for 'num_parents' times.
            min_fitness_idx = np.where(fitness == np.min(fitness))
            min_fitness_idx = min_fitness_idx[0][0]
            parents[parent_num, :] = pop[min_fitness_idx, :]
            fitness[min_fitness_idx] = 99999999999
        return parents
    
    def crossover(self, parents, offspring_size):
        offspring = np.empty(offspring_size, dtype=object)
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
            population.append(np.array(newChromosome, dtype=object))
            
        return np.array(population)
    
    def generateNewChromosome(self):
        max_depth           = int(self.mutateGene(0))
        subsample           = float(self.mutateGene(1))
        colsample_bytree    = float(self.mutateGene(2))
        gamma               = float(self.mutateGene(3))
        learningRate        = float(self.mutateGene(4))
        min_child_weight    = int(self.mutateGene(5))
        selectedFeatures    = self.mutateGene(6)
                
        return [max_depth, subsample, colsample_bytree, gamma, learningRate, min_child_weight, selectedFeatures]
 
        
    def mutateGene(self, geneIdx, init = True):
        
        # 1. learning_rate. Default = 0.3. Range: [0,1]
        # 2. n_estimators. 100 if the size of your data is high, 1000 is if it is medium-low
        # 3. max_depth. Default = 6. Increasing this value will make the model more complex and more likely to overfit. Range: [0,∞]
        # 4. subsample
        # Default = 1.
        # Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. and this will prevent overfitting. 
        # Range: (0,1]
        # 5. colsample_bytree. Default = 1. Range: (0,1]
        #    Default = 0. The larger gamma is, the more conservative the algorithm will be. Range: [0,∞]
        # 6. min_child_weight
        
        if geneIdx == 0:    # max_depth
            newGene = int
            newGene = random.randint(0,10)
            return newGene
        elif geneIdx == 1:  # subsample
            newGene = float
            newGene = random.randint(1,50) / 50
            return newGene
        elif geneIdx == 2:  # colsample_bytree
            newGene = float
            newGene = random.randint(1,50) / 50
            return newGene
        elif geneIdx == 3:  # gamma
            newGene = float
            newGene = random.randint(0,50)/50
            return newGene
        elif geneIdx == 4:  # learning rate
            newGene = float
            newGene = random.randint(0,50)/50
            return newGene
        elif geneIdx == 5:  # min_child_weight
            newGene = int
            newGene = random.randint(0,10)
            return newGene
        elif geneIdx == 6:  # selected features
            newGene = []
            newGene = np.random.randint(2, size = self.featureCount )
            return newGene
        
def PerformHyperparameterTuning(TestType,
                                x_train, 
                                y_train, 
                                weight,
                                n_estimators,
                                fold_count,
                                earlyStoppingRound,
                                inProcessFeatureSelection,
                                classificationMetric,
                                population_count, 
                                generation_count):
    
    populationCount = population_count
    generationCount = generation_count
    
    fold_count = fold_count
    number_of_mutated_genes = 4
    mutationRate = 0.5
    
    # Initialise GA object
    tuningGA = GA_HyperparameterTuning(TestType, fold_count, number_of_mutated_genes, mutationRate, n_estimators, weight, earlyStoppingRound, len(x_train.columns), classificationMetric)
    population = tuningGA.initializePopulation(populationCount)
        
    # Loop through a GA routine to select the best combination of features
    num_parents_mating = int(populationCount / 2)
    fitness_allGenerations = []             # for DEBUG
    population_allGenerations = []          # for DEBUG
    
    minFitness = []
    lastAverageFitness = float('inf')
    averageFitnessLog = []
    
    for generationIdx in range(generationCount):
        
        print("Generation %s" % generationIdx)
        # 1. Measuring the fitness of each chromosome in the population.
        fitness = tuningGA.calcPopulationFitness(x_train, y_train, population, inProcessFeatureSelection)
        
        temp = deepcopy(fitness)
        fitness_allGenerations.append(temp)
        population_allGenerations.append(population)
        
        # If average fitness no longer changes, break off the routine
        minFitness.append(float(np.min(fitness)))   
        currentAverageFitness = np.array(fitness).mean()
        averageFitnessLog.append(currentAverageFitness)
        print("--Generation %s. lastAverageFitness    = %f" % (generationIdx, lastAverageFitness) )
        print("--Generation %s, currentAverageFitness = %f" % (generationIdx, currentAverageFitness) )
        print("--Generation %s, minFitness = %f \n" % (generationIdx, float(np.min(fitness))) )
#        if abs((lastAverageFitness - currentAverageFitness)) < 0.00001:
#            print("========= converged =========")
#            print("========= converged =========")
#            print("========= converged =========")
#            break
#        lastAverageFitness = currentAverageFitness
        
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
    
    print("\n--Finishing--\n")
    
    # Calculate fitness of the final set of chromosomes
    fitness = tuningGA.calcPopulationFitness(x_train, y_train, population, inProcessFeatureSelection)
    
    min_fitness_idx = np.where(fitness == np.min(fitness))
    min_fitness_idx = min_fitness_idx[0][0]
    tunedParameters = population[min_fitness_idx, :]
    return (tunedParameters, np.min(fitness), fitness_allGenerations, population)
    
    