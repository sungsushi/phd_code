import pandas as pd


def get_downstream(pre, dataframe):
    try:
        df = dataframe[dataframe['bodyId_pre'].isin(pre)]
    except:
        df = dataframe[dataframe['bodyId_pre'] == pre]
    return df