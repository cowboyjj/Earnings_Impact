import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from copy import deepcopy

import random
import time



class GA_HyperparameterTuning:
    
    def __init__(self, fold_count, number_of_mutated_genes, mutationRate):
        self.fold_count = fold_count
        self.number_of_mutated_genes = number_of_mutated_genes
        self.mutationRate = mutationRate          
        
        # All model hyperparameters are initialised as 0
        self.kernel = 'linear'
        self.gamma = 1.0
        self.C = 1.0
        self.epsilon = 0.1
        
    def initModel(self):
        # define the final model        
        
        kernel = self.kernel
        degree = 3
        if self.kernel == 'poly2':
            kernel = 'poly'
            degree = 2
        elif self.kernel == 'poly3':
            kernel = 'poly'
            degree = 3
        elif self.kernel == 'poly4':
            kernel = 'poly'
            degree = 4
        elif self.kernel == 'poly5':
            kernel = 'poly'
            degree = 5
            
#        print("1")
        self.model = SVR(kernel = kernel, gamma = self.gamma, C = self.C, epsilon = self.epsilon, degree = degree, verbose=False, max_iter = 1000, cache_size = 3000)
#        print("2")

    
    def calcPopulationFitness(self, x_train, y_train, new_population):
        folds = KFold(n_splits = self.fold_count, shuffle = True)#, random_state = 100)
        
        fitness = []
        for i in range(new_population.shape[0]):
            chromosome = new_population[i]            
            
            # Initialise hyperparameters according to settings in the current chromosome 
            self.kernel = str(chromosome[0])
            self.gamma = float(chromosome[1])
            self.C = float(chromosome[2])
            self.epsilon = float(chromosome[3])
            
            # Calculate the k-fold validation loss as the fitness value
            val_loss = 0.0
                
            print("\n--Chromosome: %i, kernel = %s, gamma = %f, C = %f, epsilon = %f" % (i, self.kernel, self.gamma, self.C, self.epsilon) )

            idx = 0
            for train_index, validate_index in folds.split(x_train):
                   
                start_time = time.time()            
                idx = idx + 1
                
                x_train_sub = x_train.iloc[train_index, :]
                y_train_sub = y_train.iloc[train_index]            
                
                x_validate  = x_train.iloc[validate_index, :]
                y_validate  = y_train.iloc[validate_index]
                            
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
#                print("3")
                self.model.fit(x_train_sub, y_train_sub)
#                print("4")
                
                # predict
                y_validate_predicted = self.model.predict(x_validate)
                val_loss_current = mean_absolute_error(y_validate, y_validate_predicted)
                val_loss = val_loss + val_loss_current
                
                print("----Fold %s, took %f seconds, val_loss = %f" % (idx, time.time() - start_time, val_loss_current))
        
            val_loss = val_loss/self.fold_count        
            fitness.append(val_loss)
            
#        fitness = []
#        for i in range(new_population.shape[0]):
#            chromosome = new_population[i]            
#            
#            # Initialise hyperparameters according to settings in the current chromosome 
#            self.kernel = str(chromosome[0])
#            self.gamma = float(chromosome[1])
#            self.C = float(chromosome[2])
#                        
#            # Calculate the k-fold validation loss as the fitness value
#                
#            print("--Chromosome: %i, kernel = %s, gamma = %f, C = %f" % (i, self.kernel, self.gamma, self.C) )
#            
#            self.initModel()
#            #scores = cross_val_score(self.model, x_train, y_train, cv = self.fold_count, scoring = 'neg_mean_absolute_error')
#            self.model.fit(x_train, y_train)
#            print("3")
#            #fitness.append(scores.mean())
#            print("4")
#            
#            #print("averaged score: %f\n" % scores.mean())
            
        return fitness
    
        
    def select_mating_pool(self, pop, fitness, num_parents):
        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
#        parents = np.empty((num_parents, pop.shape[1]))
        parents = []
        for parent_num in range(num_parents):
            # Within each loop find the chromosome within the current generation with the smallest fitness
            # and assign this chromosome to the 'parents' list. This repeats for 'num_parents' times.
            min_fitness_idx = np.where(fitness == np.min(fitness))
            min_fitness_idx = min_fitness_idx[0][0]
#            parents[parent_num, :] = pop[min_fitness_idx, :]
            parents.append(pop[min_fitness_idx, :])
            fitness[min_fitness_idx] = 99999999999
        return np.array(parents)
    
    
    def crossover(self, parents, offspring_size):
#        offspring = np.empty(offspring_size)
        offspring = []
        # The point at which crossover takes place between two parents. Usually it is at the center.
        crossover_point = np.uint8(offspring_size[1]/2)
    
        for k in range(offspring_size[0]):
            # Index of the first parent to mate.
            parent1_idx = k%parents.shape[0]
            # Index of the second parent to mate.
            parent2_idx = (k+1)%parents.shape[0]
            # The new offspring will have its first half of its genes taken from the first parent.
#            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            a = list(parents[parent1_idx, 0:crossover_point])
            # The new offspring will have its second half of its genes taken from the second parent.
#            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
            b = list(parents[parent2_idx, crossover_point:])
            offspring.append(a + b)
        return np.array(offspring)
    
    
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
        kernal             = str(self.mutateGene(0))
        gamma              = float(self.mutateGene(1))
        C                  = float(self.mutateGene(2))
        epsilon            = float(self.mutateGene(3))
        return [kernal, gamma, C, epsilon]
    
    
    def mutateGene(self, geneIdx):
        
        if geneIdx == 0:    # kernal
            newGene = str
            newGene = random.choice(['poly2', 'poly3', 'poly4', 'poly5', 'rbf', 'sigmoid', 'linear'])
            
        elif geneIdx == 1:  # gamma
            newGene = float
            newGene = random.choice(np.logspace(-2, 4, 20))
            return newGene
        
        elif geneIdx == 2:  # C
            newGene = float
            newGene = random.choice(np.logspace(-2, 4, 20))
            return newGene
        
        elif geneIdx == 3:  # epsilon
            newGene = float
            newGene = random.choice([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
            
        return newGene
    
    
def PerformHyperparameterTuning(x_train_featureSelected, 
                                y_train_featureSelected, 
                                population_count, 
                                generation_count):
    
    populationCount = population_count
    generationCount = generation_count
    fold_count = 3
    number_of_mutated_genes = 2
    mutationRate = 0.4
    featureCount = len(x_train_featureSelected.columns)
        
    # Initialise GA object
    tuningGA = GA_HyperparameterTuning(fold_count, number_of_mutated_genes, mutationRate)
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
        fitness = tuningGA.calcPopulationFitness(x_train_featureSelected, y_train_featureSelected, population)
        
        fitness_allGenerations.append(deepcopy(fitness))
        population_allGenerations.append(deepcopy(population))
        
        # If average fitness no longer changes, break off the routine
        minFitness.append(float(np.min(fitness)))   
        currentAverageFitness = np.array(minFitness).mean()
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
    
    # Calculate fitness of the final set of chromosomes
    fitness = tuningGA.calcPopulationFitness(x_train_featureSelected, y_train_featureSelected, population)
    fitness_allGenerations.append(deepcopy(fitness))
    population_allGenerations.append(deepcopy(population))
    
    min_fitness_idx = np.where(fitness == np.min(fitness))
    min_fitness_idx = min_fitness_idx[0][0]
    tunedParameters = population[min_fitness_idx, :]
    
    return (tunedParameters, generationIdx, np.min(fitness), fitness_allGenerations, population_allGenerations)
    
    