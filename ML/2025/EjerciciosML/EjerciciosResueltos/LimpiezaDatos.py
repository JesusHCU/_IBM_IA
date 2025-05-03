import pandas as pd

def rellenar_con_media(dataframe, columna):
    media = dataframe[columna].mean()
    dataframe[columna].fillna(media, inplace=True)
    return dataframe