import pandas as pd

def seleccionar_datos(dataframe, criterio):
    return dataframe.query(criterio)