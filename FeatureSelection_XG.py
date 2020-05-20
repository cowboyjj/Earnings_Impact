import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler, StandardScaler
import xgboost as xgb
from copy import deepcopy
import random
import time



class GA_FeatureSelection:
    
    def __init__(self, testType, fold_count, number_of_mutated_genes, featureCount, mutationRate, weight):
        self.fold_count = fold_count
        self.number_of_mutated_genes = number_of_mutated_genes
        self.mutationRate = mutationRate          
        self.featureCount = featureCount
        self.testType = testType
        
        self.n_estimators = 4000
        self.earlyStoppingRound = 10
        self.max_depth = 1
        self.subsample = 0.8
        self.colsample_bytree = 0.38
        self.gamma = 0.38
        self.learningRate = 0.7
        self.min_child_weight = 1
        
#        self.n_estimators = 4000
#        self.earlyStoppingRound = 10
#        self.max_depth = 8
#        self.subsample = 0.34
#        self.colsample_bytree = 0.04
#        self.gamma = 0.84
#        self.learningRate = 0.96
#        self.min_child_weight = 2
        
        self.sample_weight = weight

    
    def calcPopulationFitness(self, x_train, y_train, featureCount, new_population):
        
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
                metrics = ['error']
            else:
                objective = 'multi:softprob'
                metrics = ['merror']
                
        fitness = []
        for i in range(new_population.shape[0]):
            
            chromosome = new_population[i]
            
            # Filter the input features according to the structure of the current chromosome
            x_train_filtered = deepcopy(x_train)
            if len(chromosome) != featureCount:
                print("error!")
                
            columnsToDrop = []
            for idx in range(featureCount):
                val = chromosome[idx]
                if val == 0:
                    columnsToDrop.append(idx)
                    
            x_train_filtered = x_train_filtered.drop(x_train_filtered.columns[columnsToDrop], axis=1)
            
            # Create training data DMatrix object
            if self.testType == "Regression":
                xg_train = xgb.DMatrix(x_train_filtered, label=y_train)
            else:
                xg_train = xgb.DMatrix(x_train_filtered, label=y_train, weight = self.sample_weight)
                                 
            
            error = 0.0
            val_error = 0.0
                
#            print("--Chromosome: %i, eta = %f, max_depth = %i, colsample_bytree = %f, subsample = %f, gamma = %f, min_child_weight = %i" % (i, self.learningRate, self.max_depth, self.colsample_bytree, self.subsample, self.gamma, self.min_child_weight) )

            # 2. Set the parameters to XGBoost            
            params = {'eta': self.learningRate, 
                      'max_depth': self.max_depth, 
                      'subsample': self.subsample, 
                      'colsample_bytree': self.colsample_bytree,
                      'gamma': self.gamma,
                      'eval_metric': metrics, 
                      'min_child_weight': self.min_child_weight,
                      'silent':1,
                      'nthread':4,
                      'scale_pos_weight':1,
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
                    val_error = list(CV['test-error-mean'])[-1]
                    error = list(CV['train-error-mean'])[-1]
                else:
                    val_error = list(CV['test-merror-mean'])[-1]
                    error = list(CV['train-merror-mean'])[-1]
                
            print("--Chromosome: %i, val_error = %f, error = %f" % (i, val_error, error))
            print("\n")
                        
            # Use Validation Loss value as the fitness value
            fitness.append(val_error)
                        
        return fitness
        
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
        
        featureCount = offspring_crossover.shape[1]
        
        for idx in range(offspring_crossover.shape[0]):
            if random.random() < self.mutationRate:
                
                print("*********** Mutating ***********")
                chromosome = offspring_crossover[idx]
                for i in range(self.number_of_mutated_genes):
                    mutateIdx = random.randint(0, featureCount-1)
                    chromosome[mutateIdx] = 1 - chromosome[mutateIdx]       # turn 0 to 1 or 1 to 0
                
                offspring_crossover[idx] = chromosome
    
        return offspring_crossover  
 
    
    
def PerformFeatureSelection(testType,
                            x_train, 
                            y_train, 
                            weight,
                            population_count, 
                            generation_count):
    
    populationCount = population_count
    generationCount = generation_count
    
    fold_count = 3
    number_of_mutated_genes = 5
    mutationRate = 0.2
    featureCount = len(x_train.columns)
    
    # Creating the initial population.
    temp = []
    for i in range(populationCount):
#        pop = np.random.randint(2, size = featureCount )
#        temp.append(np.array(pop))
        
        
        #x = int(featureCount / 2)
        x = 2
        a = [1] * x
        b = [0] * (featureCount -x)
        
        bestFeatures = []
        bestFeatures = a + b
        random.shuffle(bestFeatures)
        temp.append(np.array(bestFeatures))
    population = np.array(temp)
    
    # Initialise GA object
    featureSelectionGA = GA_FeatureSelection(testType, fold_count, number_of_mutated_genes, featureCount, mutationRate, weight)
        
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
        fitness = featureSelectionGA.calcPopulationFitness(x_train, y_train, featureCount, population)
        
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
        parents = featureSelectionGA.select_mating_pool(population, fitness, num_parents_mating)
    
        # 3. Generating next generation using crossover.
        offspring_crossover = featureSelectionGA.crossover(parents,
                                           offspring_size=(populationCount - parents.shape[0], parents.shape[1]))
    
        # 4. Adding some variations to the offsrping using mutation.
        offspring_mutation = featureSelectionGA.mutation(offspring_crossover)
        
        # Creating the new population based on the parents and offspring.
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = offspring_mutation
    
    print("\n--Finishing--\n")
    
    # Calculate fitness of the final set of chromosomes
    fitness = featureSelectionGA.calcPopulationFitness(x_train, y_train, featureCount, population)
    
    min_fitness_idx = np.where(fitness == np.min(fitness))
    min_fitness_idx = min_fitness_idx[0][0]
    bestFeatures = population[min_fitness_idx, :]
    return (bestFeatures, generationIdx, np.min(fitness), fitness_allGenerations, population)
    
    