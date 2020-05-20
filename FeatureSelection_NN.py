import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
import random
import time



class GA_FeatureSelection:
    
    def __init__(self, fold_count, epochs_count, batchSize, number_of_mutated_genes, hiddenLayerNeuronCount, featureCount):
        self.fold_count = fold_count
        self.epochs_count = epochs_count
        self.batchSize = batchSize
        self.number_of_mutated_genes = number_of_mutated_genes
        self.mutationRate = 0.04
        self.hiddenLayerNeuronCount = hiddenLayerNeuronCount
        self.featureCount = featureCount
            
        
    def initModel(self):
        # define the final model        
        
        self.model = Sequential()    
        self.model.add(Dense(self.hiddenLayerNeuronCount, input_dim = self.featureCount, activation='relu'))
        self.model.add(Dense(self.hiddenLayerNeuronCount, activation='relu'))
        self.model.add(Dense(1, activation='linear'))
        self.model.compile(loss='mse', optimizer='adam', metrics = ['mae']) # calculate additional metric 'mean absolute error'
        
    def cal_pop_fitness(self, equation_inputs, pop):
        # Calculating the fitness value of each solution in the current population.
        # The fitness function caulcuates the sum of products between each input and its corresponding weight.
        fitness = np.sum(pop*equation_inputs, axis=1)
        return fitness
    
    def calcPopulationFitness(self, x_train, y_train, featureCount, new_population):
        
        fitness = []
        lossValuePerPopulation = []
        for i in range(new_population.shape[0]):
            chromosome = new_population[i]
            
            # Filter the input features according to the structure of the current chromosome
            x_train_filtered = []
            for point in x_train:
                filtered = [i*j for i,j in zip(chromosome, point)]
                x_train_filtered.append(filtered)
                    
            # Calculate the k-fold validation loss as the fitness value
            val_loss = 0.0
            loss = 0.0
            mae = 0.0
                
            print("--Chromosome: %i" % i)
            folds = KFold(n_splits = self.fold_count, shuffle = True, random_state = 1)
            idx = 0
            for train_index, validate_index in folds.split(x_train_filtered):
                   
                start_time = time.time()            
                idx = idx + 1
                
                x_train_sub = np.array(x_train_filtered)[train_index]
                y_train_sub = np.array(y_train)[train_index]            
                
                x_validate  = np.array(x_train_filtered)[validate_index]
                y_validate  = np.array(y_train)[validate_index]

                # For each fold, standardize the training data.           
                scaler = RobustScaler()
                scaler.fit(x_train_sub)
                x_train_sub = scaler.transform(x_train_sub.tolist())
                
                # Also standardize the validation data using the same scaler.      
                x_validate = scaler.transform(x_validate.tolist())
            
                # Create a new model for each fold
                self.initModel()
                
                # fit the final model
                fitResult = self.model.fit(x_train_sub, 
                                           y_train_sub, 
                                           epochs = self.epochs_count, 
                                           batch_size = self.batchSize, 
                                           verbose = 0, 
                                           validation_data=(x_validate, y_validate))
            
                current_val_loss = fitResult.history["val_loss"][-1]
                current_loss = fitResult.history["loss"][-1]
                current_mae = fitResult.history["mean_absolute_error"][-1]
                
                val_loss = val_loss + current_val_loss
                loss = loss + current_loss
                mae = mae + current_mae
                
                print("----Fold %s, took %f seconds, val_loss = %f, loss = %f, mae = %f" % (idx, time.time() - start_time, current_val_loss, current_loss, current_mae))
        
            val_loss = val_loss/self.fold_count        
            loss = loss / self.fold_count
            print("--Chromosome: %i, ave(val_loss) = %f, ave(loss) = %f, ave(mae) = %f" % (i, val_loss, loss, mae))
            
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
              
    
def PerformFeatureSelection(x_train, y_train, featureCount, batch_size, epochs_count, hiddenLayerNeuronCount, population_count, generation_count, numberOfMutatedGenes, checkConvergence = True):
    
    populationCount = population_count
    generationCount = generation_count
    number_of_mutated_genes = numberOfMutatedGenes
    fold_count = 5    
    batchSize = batch_size
    
    # Creating the initial population.
    temp = []
    for i in range(populationCount):
        pop = np.random.randint(2, size = featureCount )
        temp.append(np.array(pop))
    population = np.array(temp)
        
    # Initialise GA object
    featureSelectionGA = GA_FeatureSelection(fold_count, epochs_count, batchSize, number_of_mutated_genes, hiddenLayerNeuronCount, featureCount)
#    featureSelectionGA.initModel(hiddenLayerNeuronCount, featureCount)
        
    # Loop through a GA routine to select the best combination of features
    num_parents_mating = int(populationCount / 2)
    fitness_allGenerations = []             # for DEBUG
    lossValue_allGenerations = []           # for DEBUG
    population_allGenerations = []          # for DEBUG
    
    minFitness = []
    lastAverageFitness = float('inf')
    
    for generationIdx in range(generationCount):
        
        print("Generation %s" % generationIdx)
        # 1. Measuring the fitness of each chromosome in the population.
        ret = featureSelectionGA.calcPopulationFitness(x_train, y_train, featureCount, population)
        
        fitness = ret[0]
        loss = ret[1]
        temp = fitness
        fitness_allGenerations.append(temp)
        lossValue_allGenerations.append(loss)
        population_allGenerations.append(population)
    
        # If average fitness no longer changes, break off the routine
        minFitness.append(float(np.min(fitness)))   
        currentAverageFitness = np.array(fitness).mean()
        print("--Generation %s. lastAverageFitness    = %f" % (generationIdx, lastAverageFitness) )
        print("--Generation %s, currentAverageFitness = %f \n" % (generationIdx, currentAverageFitness) )
        if checkConvergence == True and abs((lastAverageFitness - currentAverageFitness)) < 0.00001:
            print("========= converged =========")
            print("========= converged =========")
            print("========= converged =========")
            break
        lastAverageFitness = currentAverageFitness
        
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
    
    # Calculate fitness of the final set of chromosomes
    fitness = tuningGA.calcPopulationFitness(x_train_featureSelected, y_train_featureSelected, population)
    
    min_fitness_idx = np.where(fitness == np.min(fitness))
    min_fitness_idx = min_fitness_idx[0][0]
    bestFeatures = population[min_fitness_idx, :]
    return (bestFeatures, generationIdx, currentAverageFitness)
    
    