import pickle
import sys
from multiprocessing import Pool
import csv
import time

from genetic import FloatGene, Genome, IntGene, Population, StringGene, GeneSequence

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
import numpy as np

np.random.seed(0)#seta seed para reproducibilidade

POPULATION_PICKLE_FILE = 'population.pkl'
BEST_NN_PICKLE_FILE = 'neural_model.pkl'
CSV_FILE = 'log_results.csv'

POPULATION_SIZE = 60
SELECTION_SIZE = 20
MUTATION_RATE = .01

N_GENERATIONS = 1000

def mse_to_fitness(mse):
    fitness = 1. / (mse + .00001)
    return fitness

def genome_to_NN(genome: Genome):
    act = genome.genes[0].getValue()
    slv = genome.genes[1].getValue()
    lrs = genome.genes[2].getValue()
    lr = genome.genes[3].getValue()
    hls = genome.genes[4].getValue()
    hls = tuple(hls)

    clf = MLPRegressor(hidden_layer_sizes=hls, activation=act,
                       solver=slv, learning_rate=lrs, learning_rate_init=lr, random_state=0, max_iter=400)
    return clf

#carrega arquivo e cria dados de treinamento e teste
#necessário fazer fora do main para passar pra função rate_nn
#pois será executada em paralelo
df = pd.read_csv('data.csv')

X = df.iloc[:50000, 5:]
y = df.iloc[:50000, 0:5]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)

def rate_nn(nn):
    nn.fit(X_train, y_train)
    y_pred = nn.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    fitness = mse_to_fitness(mse)
    return fitness
    
def main():


    if sys.argv[1] == 'reset':
        print('População reinicializada')

        #inicializa genes e genoma
        hiddenLayerGene = IntGene(
            initial_value=10, min_value=5, max_value=150, deviation=5)
        architectureGene = GeneSequence(
            genes=[hiddenLayerGene, hiddenLayerGene], minGenes=1, maxGenes=20, change_size_prob=.5)
        activationGene = StringGene(
            values_list=['relu', 'identity', 'tanh', 'logistic'], initial_value_index=0)
        solverGene = StringGene(
            values_list=['lbfgs', 'sgd', 'adam'], initial_value_index=2)
        learningRateStrategyGene = StringGene(
            values_list=['constant', 'invscaling', 'adaptive'], initial_value_index=0)
        learningRateGene = FloatGene(
            initial_value=.001, min_value=.0001, max_value=.01, deviation=.0005)

        genome = Genome(genes=[activationGene, solverGene,
                        learningRateStrategyGene, learningRateGene, architectureGene])

        #inicia primeira população com genomas iguais e os randomiza depois
        population = Population(genomes=[genome] * POPULATION_SIZE,
                                n_selected=SELECTION_SIZE, mutation_rate=MUTATION_RATE, crossover_strategy='locus')
        population.randomize_population(n_generations=500)

    if sys.argv[1] == 'load':
        print('População carregada')
        with open(POPULATION_PICKLE_FILE, 'rb') as pkl_file:
            population = pickle.load(pkl_file)

    print('População inicial:')
    print(population)
    

    pool = Pool()

    best_fitness = 0

    try:
        start = time.time()
        for i in range(N_GENERATIONS):

            nns = [genome_to_NN(genome) for genome in population.genomes]
            fitness_list = pool.map(rate_nn, nns)

            population.nextGeneration(fitness_list)
            print(f'GERAÇÃO {i+1}')
            print(population)
            #salva população atual na memória
            with open(POPULATION_PICKLE_FILE, 'wb') as pfile:
                pickle.dump(population, pfile, protocol=pickle.HIGHEST_PROTOCOL)
            #mantém histórico de fitness
            with open(CSV_FILE, 'a+', newline='') as write_obj:
                writer = csv.writer(write_obj)
                writer.writerow(fitness_list)

            #caso obtenha uma rede neural com melhor desempenho, a salva
            index_best_nn = np.argmax(fitness_list)
            if fitness_list[index_best_nn] > best_fitness:
                best_nn = nns[index_best_nn]
                best_fitness = fitness_list[index_best_nn]
                with open(BEST_NN_PICKLE_FILE, 'wb') as pfile:
                    pickle.dump(best_nn, pfile, protocol=pickle.HIGHEST_PROTOCOL)
                print(f'Nova melhor rede neural com fitness = {best_fitness:.5}:')
                print(best_nn)

        end = time.time()
        print(f'{end-start}')
    except KeyboardInterrupt:
        print('Interrompido pelo usuário')
        quit()



if __name__ == '__main__':
    main()


