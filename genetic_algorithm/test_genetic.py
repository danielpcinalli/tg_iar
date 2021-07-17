from math import log10
import util

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
    l = util.locus(
        list1=l1,
        list2=l2,
        locus=2
    )

    assert l == [2, 's', 'c', 0, 'e']

def test_select_indexes():
    l = util.select_indexes(m=5, n=1, probs=[1., 0, 0, 0, 0])

    assert l == [0]