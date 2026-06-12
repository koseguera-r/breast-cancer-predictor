import pandas as pd

# Cargar el dataset
df = pd.read_csv('wdbc_dataset.csv')

# Mostrar primeras 5 filas
print("Primeras 5 filas:")
print(df.head())

# Mostrar información general
print("\nInformación del dataset:")
print(df.info())

# Ver nombres de columnas
print("\nNombres de columnas:")
print(df.columns.tolist())