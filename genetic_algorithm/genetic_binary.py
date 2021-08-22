import numpy as np
from typing import List
import util
import logging
logging.basicConfig(filename='logs.log', level=logging.INFO)

#população de n cromossomos binários de tamanho m
#mutação e 

class Population:
    def __init__(self, population_size, cromossome_size, n_selected, mutation_rate=0.01) -> None:
        self.cromossomes = [util.random_binary_cromossome(size=cromossome_size) for _ in range(population_size)]
        self.n = n_selected
        self.population_size = population_size
        self.cromossome_size = cromossome_size
        self.mutation_rate = mutation_rate
        self.generation = 1


    def nextGeneration(self, fitness_list: List[float]):
        """
        Dado uma lista com o fitness de cada genoma, 
        gera uma nova população
        """
        self.generation += 1
        logging.info(f'geração {self.generation}')
        selected_cromossomes, selected_probabilities = self.selection(fitness_list)
        
        
        self.cromossomes = []
        self.cromossomes.extend(selected_cromossomes)  # estratégia elitista

        while len(self.cromossomes) < self.population_size:

            genome1_index, genome2_index = util.select_indexes(2, len(selected_cromossomes), selected_probabilities)
            
            newCromossomes = self.crossover(genome1_index, genome2_index)
            logging.info(f'-Novos genomas inseridos: {newCromossomes[0]}')
            self.cromossomes.extend(newCromossomes)
        #como dois genomas são inseridos de cada vez, é possível ultrapassar a quantidade de genomas caso self.n seja ímpar
        self.cromossomes = self.cromossomes[0:self.population_size]
        self._mutate()

    def crossover_locus(self, cromossome_index1, cromossome_index2):
        locus = np.random.random_integers(low=1, high=self.cromossome_size-1)
        cromossomes1 = self.cromossomes[cromossome_index1]
        cromossomes2 = self.cromossomes[cromossome_index2]
        newCromossome1 = util.binary_locus(cromossomes1, cromossomes2, locus)
        newCromossome2 = util.binary_locus(cromossomes2, cromossomes1, locus)
        return [newCromossome1, newCromossome2]

    def _mutate(self):
        for cromossome_index in range(self.population_size):
            for gene_index in range(self.cromossome_size):
                if util.event_with_probability(self.mutation_rate):
                    self.cromossomes[cromossome_index] = util.flip_bit_at(self.cromossomes[cromossome_index, gene_index])


    
    def selection(self, fitness_list):
        probabilities = util.weights_to_probability(fitness_list)

        selected_indexes = util.select_indexes(self.n, self.population_size, probabilities)
        selected_cromossomes = np.array(self.cromossomes)[selected_indexes]
        
        selected_probabilities = probabilities[selected_indexes]
        selected_probabilities = util.weights_to_probability(selected_probabilities)
        selected_cromossomes = [genome.copy() for genome in selected_cromossomes]
        return selected_cromossomes, selected_probabilities

    def randomize_population(self, n_generations):

        for _ in range(n_generations):
            self._mutate()

    def __repr__(self) -> str:
        r = ''
        for genome in self.cromossomes:
            r += repr(genome) + '\n'
        return r




def decode_binary_cromossome(bc):
    LAYER_SIZE_SIZE = 10
    LAYERS_SIZE = 4
    SOLVER_SIZE = 4
    LEARNING_RATE_SIZE = 10
    LR_STRATEGY = 4
    ACTIVATION_SIZE = 4


    #pega cada pedaço do cromossomo para sua variável
    lsz_c, bc = util.consume_string(bc, LAYER_SIZE_SIZE)
    lys_c, bc = util.consume_string(bc, LAYERS_SIZE)
    slv_c, bc = util.consume_string(bc, SOLVER_SIZE)
    lnr_c, bc = util.consume_string(bc, LEARNING_RATE_SIZE)
    lrs_c, bc = util.consume_string(bc, LR_STRATEGY)
    act_c, bc = util.consume_string(bc, ACTIVATION_SIZE)

    lsz_c = util.bin_string_to_decimal(lsz_c)
    lys_c = util.bin_string_to_decimal(lys_c)
    slv_c = util.bin_string_to_decimal(slv_c)
    lnr_c = util.bin_string_to_decimal(lnr_c)
    lrs_c = util.bin_string_to_decimal(lrs_c)
    act_c = util.bin_string_to_decimal(act_c)

    lsz = util.compress_uniform(lsz_c, 10, 200, 0, 2**LAYER_SIZE_SIZE - 1)
    lys = util.compress_uniform(lys_c, 1, 10, 0, 2**LAYERS_SIZE - 1)
    slv = util.compress_uniform(slv_c, 0, 2, 0, 2**SOLVER_SIZE - 1)
    lnr = util.compress_uniform(lnr_c, .0001, .01, 0, 2**LEARNING_RATE_SIZE - 1)
    lrs = util.compress_uniform(lrs_c, 0, 2, 0, 2**LR_STRATEGY - 1)
    act = util.compress_uniform(act_c, 0, 3, 0, 2**ACTIVATION_SIZE - 1)

    d = dict({  'layer_size': int(lsz), 
                'layers': int(lys),
                'solver': ['lbfgs', 'sgd', 'adam'][int(slv)],
                'learning_rate': lnr,
                'learning_rate_strategy': ['constant', 'invscaling', 'adaptive'][int(lrs)],
                'activation': ['relu', 'identity', 'tanh', 'logistic'][int(act)]})

    return d