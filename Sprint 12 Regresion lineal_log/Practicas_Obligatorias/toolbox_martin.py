# TOOLBOX MARTIN

import pandas as pd
import numpy as np


def analizar_df(dataframe):
    # Inicializar listas para almacenar información
    columnas = []
    dtypes = []
    cardinalidades = []
    porcentajes = []

    # Iterar sobre las columnas del DataFrame
    for col in dataframe.columns:
        # Obtener el tipo de dato de la columna
        dtype = dataframe[col].dtype

        # Obtener la cardinalidad y porcentaje
        unique_values = dataframe[col].nunique()
        total_values = len(dataframe[col])
        porcentaje = (unique_values / total_values) * 100

        # Almacenar la información en las listas
        columnas.append(col)
        dtypes.append(dtype)
        cardinalidades.append(unique_values)
        porcentajes.append(porcentaje)

    # Crear un DataFrame con la información recopilada
    resultados = pd.DataFrame({
        'Columna': columnas,
        'Tipo': dtypes,
        'Cardinalidad': cardinalidades,
        'Porcentaje de valores únicos': porcentajes
    })

    return resultados