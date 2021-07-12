import numpy as np

def between(value, minValue, maxValue):
    value = max(value, minValue)
    value = min(value, maxValue)
    return value

def event_with_probability(p):
    if np.random.random() <= p:
        return True
    return False