from numpy import testing
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from genetic import FloatGene, Genome, IntGene, Population, StringGene, GeneSequence
import pickle
import sys
import numpy as np
np.random.seed(0)

PICKLE_FILE = 'population.pkl'

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




def main():
    #carrega arquivo e cria dados de treinamento e teste
    df = pd.read_csv('data.csv')

    X = df.iloc[:15000, 5:]
    y = df.iloc[:15000, 0:5]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)

    if sys.argv[1] == 'reset':
        print('População reinicializada')

        #inicializa genes e genoma
        hiddenLayerGene = IntGene(
            initial_value=20, min_value=5, max_value=80, deviation=5)
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
        population = Population(genomes=[genome] * 20,
                                n_selected=8, mutation_rate=.1, crossover_strategy='mix')
        population.randomize_population(n_generations=500)
    if sys.argv[1] == 'load':
        print('População carregada')
        with open(PICKLE_FILE, 'rb') as pkl_file:
            population = pickle.load(pkl_file)

    print('População inicial:')
    print(population)


    n_generations = 10

    try:

        for i in range(n_generations):
            fitness_list = []
            for genome in population.genomes:
                nn = genome_to_NN(genome)
                nn.fit(X_train, y_train)
                y_pred = nn.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                fitness = mse_to_fitness(mse)
                fitness_list.append(fitness)

            population.nextGeneration(fitness_list)
            print(f'GERAÇÃO {i+1}')
            print(population)
            #salva população atual na memória
            with open(PICKLE_FILE, 'wb') as pfile:
                pickle.dump(population, pfile, protocol=pickle.HIGHEST_PROTOCOL)

    except KeyboardInterrupt:
        print('Interrompido pelo usuário')
        quit()

if __name__ == '__main__':
    main()