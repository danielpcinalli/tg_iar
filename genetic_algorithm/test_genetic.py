import util

def test_mix():
    l = util.mix(
        list1=[2, 's', (1,2), 'a', ['a', 'b']],
        list2=['a', 1, 'c', 0, 'e',],
        list_bool=[True, False, True, False, True]
    )

    assert l == [2, 1, (1,2), 0, ['a', 'b']]