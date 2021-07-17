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

def locus(list1: List, list2: List, locus):
    newList = list1[0:locus] + list2[locus:]

    return newList
