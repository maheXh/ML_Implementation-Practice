import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def remove_outliear(df, col, l, u):
    q1 = df[col].quantile(l)
    q3 = df[col].quantile(u)

    IQR = q3 - q1

    lower = q1 - 1.5 * IQR
    upper = q3 + 1.5 * IQR

    df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df


def normalize(df, col):
    return (df[col] - np.mean(df[col])) / np.std(df[col])


def clean_data(df):
    df = remove_outliear(df, "Pregnancies", 0.25, 0.75)
    df = remove_outliear(df, "Glucose", 0.25, 0.75)
    df = remove_outliear(df, "BloodPressure", 0.25, 0.85)
    df = remove_outliear(df, "SkinThickness", 0.25, 0.75)
    df = remove_outliear(df, "Insulin", 0.1, 0.9)
    df = remove_outliear(df, "BMI", 0.25, 0.85)
    df = remove_outliear(df, "DiabetesPedigreeFunction", 0.25, 0.75)
    column_names = df.columns.tolist()
    column_names.pop(-1)
    for i in column_names:
        df[i] = normalize(df, i)
    return df
