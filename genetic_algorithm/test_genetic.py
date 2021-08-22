from math import log10
import util
import numpy as np

l1 = [2,   's', (1,2),  'a',  ['a', 'b']]
l2 = ['a',  1,   'c',    0,      'e',]
def test_mix():
    l = util.mix(
        list1=l1,
        list2=l2,
        list_bool=[True, False, True, False, True]
    )

    assert l == [2, 1, (1,2), 0, ['a', 'b']]

def test_locus():
    l = util.locus_crossover(
        list1=l1,
        list2=l2,
        locus=2
    )

    assert l == [2, 's', 'c', 0, 'e']

def test_select_indexes1():
    
    l = util.select_indexes(m=5, n=1, probs=[1., 0, 0, 0, 0])

    assert l == [0]

def test_select_indexes2():
    M = 20
    N = 5
    l = util.select_indexes(m=M, n=N)

    assert len(l) == N
    assert min(l) >= 0
    assert max(l) < M

def test_select_indexes3():
    #checa funcionamento quando probs = None
    l = util.select_indexes(n=5, m=20)

def test_fitness_to_probability():
    fitness_list = [0, 1, 2, 3, 4]
    probs = util.weights_to_probability(fitness_list)
    
    assert (probs == np.array([0, 1/10, 2/10, 3/10, 4/10])).all()

def test_mse_to_fitness():
    mse_list = [10, 5, 2, 1, .7, .5, .1, 0]

    fitness_list = [util.mse_to_fitness(mse) for mse in mse_list]
    assert fitness_list == sorted(fitness_list)

def test_r2_to_fitness():
    r2_score_list = [-50, -20, -10, -5, -.5, 0, .5, .7, 1]

    fitness_list = [util.r2_to_fitness(r2) for r2 in r2_score_list]
    assert fitness_list == sorted(fitness_list)

def test_bin_flip_bit():
    b = '00001000'
    b = util.flip_bit_at(b, 3)
    b = util.flip_bit_at(b, 4)

    assert b == '00010000'

def test_bin_locus_crossover():
    c1 = '000000'
    c2 = '111111'
    c = util.binary_locus(c1, c2, 2)

    assert c == '001111'

def test_bin_string_to_decimal():
    s = '100'
    d = util.bin_string_to_decimal(s)

    assert d == 4

def test_consume_string():
    s = '0010111001000'
    x, s = util.consume_string(s, 5)

    assert x == '00101'
    assert s == '11001000'

def test_compress_uniform():
    n = util.compress_uniform(500, 0, 100, 400, 600)
    assert int(n) == 50
