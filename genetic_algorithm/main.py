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

PICKLE_FILE = 'population.pkl'
CSV_FILE = 'log_results.csv'

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

X = df.iloc[:20000, 5:]
y = df.iloc[:20000, 0:5]
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
            initial_value=5, min_value=5, max_value=40, deviation=5)
        architectureGene = GeneSequence(
            genes=[hiddenLayerGene, hiddenLayerGene], minGenes=1, maxGenes=10, change_size_prob=.5)
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
        population = Population(genomes=[genome] * 30,
                                n_selected=8, mutation_rate=.05, crossover_strategy='locus')
        population.randomize_population(n_generations=1000)

    if sys.argv[1] == 'load':
        print('População carregada')
        with open(PICKLE_FILE, 'rb') as pkl_file:
            population = pickle.load(pkl_file)

    print('População inicial:')
    print(population)
    n_generations = 100

    pool = Pool()

    try:
        start = time.time()
        for i in range(n_generations):

            nns = [genome_to_NN(genome) for genome in population.genomes]
            fitness_list = pool.map(rate_nn, nns)

            population.nextGeneration(fitness_list)
            print(f'GERAÇÃO {i+1}')
            print(population)
            #salva população atual na memória
            with open(PICKLE_FILE, 'wb') as pfile:
                pickle.dump(population, pfile, protocol=pickle.HIGHEST_PROTOCOL)
            with open(CSV_FILE, 'a+', newline='') as write_obj:
                writer = csv.writer(write_obj)
                writer.writerow(fitness_list)
        end = time.time()
        print(f'{end-start}')
    except KeyboardInterrupt:
        print('Interrompido pelo usuário')
        quit()



if __name__ == '__main__':
    main()


