import os
import pandas as pd

def extraer_categorias(df):
    descripciones_unicas = df.DescripcionEnEstudio.unique()



def get_categorias():
    if os.path.exists("categorias.csv"):
        df = pd.read_csv("categorias.csv")
        return df
        pass #cargar categorias
