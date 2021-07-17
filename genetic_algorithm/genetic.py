import numpy as np
from typing import List, Union
from numpy.lib import utils

from numpy.lib.function_base import select
import util


class Gene:
    def __init__(self):
        pass

    def mutate(self):
        pass

    def __repr__(self) -> str:
        return f"{self.value}"

    def getValue(self):
        return self.value


class IntGene(Gene):
    def __init__(self, initial_value, min_value, max_value, deviation):
        self.value = initial_value
        self.min = min_value
        self.max = max_value
        self.deviation = deviation

    def mutate(self):
        newValue = np.random.normal(loc=self.value, scale=self.deviation)
        newValue = max(self.min, newValue)
        newValue = min(self.max, newValue)
        return IntGene(int(newValue), self.min, self.max, self.deviation)

    def copy(self):
        return IntGene(self.value, self.min, self.max, self.deviation)

class FloatGene(Gene):
    def __init__(self, initial_value, min_value, max_value, deviation):
        self.value = initial_value
        self.min = min_value
        self.max = max_value
        self.deviation = deviation

    def mutate(self):
        newValue = np.random.normal(loc=self.value, scale=self.deviation)
        newValue = max(self.min, newValue)
        newValue = min(self.max, newValue)
        return FloatGene(newValue, self.min, self.max, self.deviation)

    def __repr__(self) -> str:
        return f"{float(self.value):.8}"

    def copy(self):
        return FloatGene(self.value, self.min, self.max, self.deviation)

class StringGene(Gene):
    def __init__(self, values_list, initial_value_index=0):
        self.values = values_list
        self.value = self.values[initial_value_index]
        self.valueIndex = initial_value_index

    def mutate(self):
        self.valueIndex = np.random.choice(len(self.values))
        return StringGene(self.values, self.valueIndex)

    def copy(self):
        return StringGene(self.values, self.valueIndex)

class GeneSequence(Gene):
    def __init__(self, genes: List[Gene], minGenes=1, maxGenes=5, change_size_prob=.5):
        self.genes = genes
        self.change_size_prob = change_size_prob
        self.minGenes = minGenes
        self.maxGenes = maxGenes

    def _change_size(self):
        # decide se irá crescer ou diminuir a sequência de genes
        increase_size = util.event_with_probability(.5)
        # garante mudar tamanho dentro dos limites
        if len(self.genes) == self.maxGenes:
            increase_size = False
        if len(self.genes) == self.minGenes:
            increase_size = True

        if increase_size:
            newGeneSequence = [gene.copy() for gene in self.genes]
            newGeneSequence.append(newGeneSequence[-1].mutate())
        else:
            newGeneSequence = [gene.copy() for gene in self.genes]
            newGeneSequence = newGeneSequence[0: -1]
        return GeneSequence(newGeneSequence, self.minGenes, self.maxGenes, self.change_size_prob)

    def mutate(self):
        # decide se irá mutar um dos genes ou se vai alterar o tamanho da sequência
        if util.event_with_probability(self.change_size_prob):
            return self._change_size()
        else:
            index = np.random.randint(0, len(self.genes))
            newGeneSequence = [gene.copy() for gene in self.genes]
            newGeneSequence[index] = newGeneSequence[index].mutate()
            return GeneSequence(newGeneSequence, self.minGenes, self.maxGenes, self.change_size_prob)

    def getValue(self):
        return [gene.getValue() for gene in self.genes]

    def copy(self):
        return GeneSequence([gene.copy() for gene in self.genes], self.minGenes, self.maxGenes, self.change_size_prob)

    def __repr__(self) -> str:
        r = ''
        for gene in self.genes:
            r += repr(gene) + ';'
        
        return r[:-1]#retira último ';'


class Genome:
    def __init__(self, genes: List[Gene]):
        self.genes = [gene.copy() for gene in genes]
        self.genomeSize = len(genes)

    def getGenes(self):
        return self.genes

    def mutate(self, mutation_rate):

        for index in range(self.genomeSize):
            if util.event_with_probability(mutation_rate):
                self.genes[index] = self.genes[index].mutate()


    @property
    def size(self):
        return len(self.genes)

    def copy(self):
        return Genome(self.genes)

    def __repr__(self) -> str:
        r = ""
        for gene in self.genes:
            r += f'({repr(gene)})'

        return r


class Population:
    def __init__(self, genomes: List[Genome], n_selected, mutation_rate=.01, crossover_strategy = 'locus'):
        self.genomes : List[Genome] = [genome.copy() for genome in genomes]
        self.n = n_selected
        self.population_size = len(self.genomes)
        self.genome_size = self.genomes[0].size
        self.mutation_rate = mutation_rate
        self.generation = 1

        if crossover_strategy == 'locus':
            self.crossover = self.crossover_locus
        if crossover_strategy == 'mix':
            self.crossover = self.crossover_mix

    def nextGeneration(self, fitness_list: List[float]):
        """
        Dado uma lista com o fitness de cada genoma, 
        gera uma nova população
        """
        self.generation += 1

        
        selected_genomes = self.selection(fitness_list)
        
        self.genomes = []
        self.genomes.extend(selected_genomes)  # estratégia elitista

        while len(self.genomes) < self.population_size:
            genome1_index, genome2_index = np.random.choice(
                a=range(len(selected_genomes)),
                size=2,
                replace=False)
            
            newGenomes = self.crossover(
                selected_genomes[genome1_index],
                selected_genomes[genome2_index])
            self.genomes.extend(newGenomes)
        #como dois genomas são inseridos de cada vez, é possível ultrapassar a quantidade de genomas caso self.n seja ímpar
        self.genomes = self.genomes[0:self.population_size]
        self._mutate()

    def crossover_locus(self, genome1: Genome, genome2: Genome):
        locus = np.random.random_integers(low=1, high=self.genome_size-1)
        genome1_genes = genome1.getGenes()
        genome2_genes = genome2.getGenes()
        newGenome1 = Genome(util.locus(genome1_genes, genome2_genes, locus))
        newGenome2 = Genome(util.locus(genome2_genes, genome1_genes, locus))

        return [newGenome1, newGenome2]

    def _mutate(self):
        for genome in self.genomes:
            genome.mutate(self.mutation_rate)
    
    def crossover_mix(self, genome1: Genome, genome2: Genome):
        genes1 = genome1.getGenes()
        genes2 = genome2.getGenes()
        
        bool_list = np.random.choice(a=[True, False], size=self.genome_size)
        newGenome1 = Genome(util.mix(genes1, genes2, bool_list))
        newGenome2 = Genome(util.mix(genes2, genes1, bool_list))
        return [newGenome1, newGenome2]

    def selection(self, fitness_list):
        probabilities = np.array(fitness_list) / sum(fitness_list)
        selected_genomes = np.random.choice(
            a=self.genomes, size=self.n, p=probabilities)
        selected_genomes = [genome.copy() for genome in selected_genomes]
        return selected_genomes

    def randomize_population(self, n_generations):

        for _ in range(n_generations):
            self._mutate()

    def __repr__(self) -> str:
        r = ''
        for genome in self.genomes:
            r += repr(genome) + '\n'
        return r
