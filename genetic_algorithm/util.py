from typing import List
import numpy as np

def between(value, minValue, maxValue):
    value = max(value, minValue)
    value = min(value, maxValue)
    return value

def event_with_probability(p):
    if np.random.random() <= p:
        return True
    return False

def mix(list1: List, list2: List, list_bool: List):
    newList = []
    for l1, l2, b in zip(list1, list2, list_bool):
        if b:
            newList.append(l1)
        else:
            newList.append(l2)
    return newList

def locus_crossover(list1: List, list2: List, locus):
    newList = list1[0:locus] + list2[locus:]

    return newList

def select_indexes(n, m, probs=None):
    """
    Retorna n índices únicos entre [0, m) de acordo com a probabilidade
    ou uniformemente caso não se passe probabilidades
    """
    try:
        return np.random.choice(a=range(m), size=n, p=probs, replace=False)
    except:
        return np.random.choice(a=range(m), size=n, replace=False)    



#x[[0,1]]
#x[[True, False, True]]

def weights_to_probability(weights: List[float]):
    probabilities = np.array(weights) / sum(weights)
    return probabilities

def mse_to_fitness(mse):
    fitness = 1. / (mse + .00001)**3
    return fitness

def r2_to_fitness(r2):
    #r2 <= 1 ; quanto mais próximo de 1 melhor
    distance_to_1 = 1 - r2
    fitness = 1. / (distance_to_1 + .00001)**3
    return fitness



def random_binary_cromossome(size):
    bin_cromossome = np.random.choice(a=['0', '1'], size=size)
    bin_cromossome = ''.join(bin_cromossome)
    return bin_cromossome

def flip_bit_at(bin_cromossome, index):
    def flip_bit(c):
        if c == '1':
            return '0'
        return '1'

    new_cromossome = bin_cromossome[0:index] + flip_bit(bin_cromossome[index]) + bin_cromossome[index+1:]

    return new_cromossome

def binary_locus(bin_cromossome1, bin_cromossome2, locus):
    
    return bin_cromossome1[:locus] + bin_cromossome2[locus:]

def bin_string_to_decimal(s):
    return int(s, 2)

def consume_string(s, n):
    return s[:n], s[n:]

def compress_uniform(number, from_n, to_n, min_n, max_n):

    number = number - min_n
    number = number / (max_n - min_n) * (to_n - from_n)
    number = number + from_n
    return number
