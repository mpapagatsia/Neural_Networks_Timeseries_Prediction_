import pandas as pd
import math
import numpy as np
threshold = 20

def limit_dataset():

    df = pd.read_csv("data_set.csv", sep='\t', encoding='utf-8')

    #drop the year 2010
    df = df.drop(df.index[54])

    #limit the data set on 12/15 valid values
    for column in df:
        last_years = df[column].tail(15).sum(axis=0, min_count=12) #7 -> 2

        if last_years < threshold or math.isnan(last_years):
            df = df.drop(column, axis=1)

    #keep the last 15 years
    for i in range(39):
        df = df.drop(df.index[0])

    df = df.drop(df.columns[[0,1]], axis=1)

    #fill the nan values with zeros
    df = df.fillna(0)

    df.to_csv("inputs.csv", sep='\t', encoding='utf-8')

def more_limitation():
    df = pd.read_csv("inputs.csv", sep='\t', encoding='utf-8')

    df = df.drop(df.columns[[0,1]], axis=1)

    df = df.fillna(0)

    #drop papers that have large variations between successive years
    labels = []

    for column in df:
        for i in range(len(df[column])-1):

            diff = abs(df[column][i] - df[column][i+1])

            if diff > 9:
                labels.append(column)
                break

    df = df.drop(labels, axis=1)

    df.to_csv("inputs.csv", sep='\t', encoding='utf-8')

limit_dataset()
more_limitation()
