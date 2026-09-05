import numpy as np

def probability_weighted_prediction(high_probability, normal_prediction, high_prediction):
    return ((1-high_probability)*normal_prediction)+(high_probability*high_prediction)

def hard_switch_prediction(high_probability, normal_prediction, high_prediction, threshold=0.5):
    return np.where(high_probability>=threshold, high_prediction, normal_prediction)
