import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def z(w, b, x):
    z_value = np.dot(w, x) + b
    return z_value


def sigmoid(w, b, x):
    z_value = z(w, b, x)
    pred = 1 / (1 + np.exp(-z_value))
    return pred
