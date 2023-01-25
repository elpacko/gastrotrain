import os
from csv import reader

import pandas as pd
import progressbar
import re
from settings import get_settings
from nltk import edit_distance

app_settings = get_settings()
capturas_root = app_settings["default"]["CAPTURAS_ROOT"]
capturas_datasource = os.path.join(app_settings["default"]["CAPTURAS_ROOT"], app_settings["default"]["DATASOURCE"])
capturas_localizaciones = os.path.join(app_settings["default"]["CAPTURAS_ROOT"], app_settings["default"]["LOCALIZACIONES"])
progressbar_settings = [progressbar.Bar("=", "[", "]"), " ", progressbar.Percentage()]


def __init__():
    pass


def get_capturas_df():
    df = pd.read_csv(capturas_datasource)
    return df

def get_capturas_localizaciones_df():
    df = pd.read_csv(capturas_localizaciones)
    return df

def init_categorias_folders():
    print("Creando folders de categorias")
    localizaciones_df = get_capturas_localizaciones_df()
    for _, localizacion_categoria in localizaciones_df.iterrows():
        categoria_path = os.path.join(capturas_root,"EstudiosSeparados",str(localizacion_categoria["Categoria"]))
        crear_directorio(categoria_path)

def get_categoria(localizaciones_df, localizacion):
    localizacion = re.sub(r'\W+', '', localizacion)
    exact_match = localizaciones_df.loc[localizaciones_df["Nombre"]==localizacion]
    if exact_match.size == 1:
        return exact_match.item()
    match_dict = {}
    for _, localizacion_categoria in localizaciones_df.iterrows():
        match_dict[localizacion_categoria["Categoria"]] = edit_distance(localizacion_categoria["Nombre"].upper(), localizacion.upper())
    min_key = min(match_dict, key=match_dict.get)
    return min_key


def crear_directorio(path):
    if (not os.path.exists(path)):
        os.mkdir(path)
    
def get_file_exists():
    print("Identificando imagenes existentes")
    capturas_df = get_capturas_df()
    file_exists_status = []
    bar = progressbar.ProgressBar(
        maxval=capturas_df.shape[0],
        widgets=progressbar_settings,
    )
    bar.start()
    for index, row in capturas_df.iterrows():
        file_name = capturas_root + row["url"]
        file_exists = os.path.isfile(file_name)
        bar.update(index)
        file_exists_status.append(file_exists)
    capturas_df["file_exists_status"] = file_exists_status
    bar.finish()
    capturas_df.to_csv(
        "file_exists_status.csv", index=False, encoding="utf-8"
    )  # False: not include index
    capturas_df["file_exists_status"].value_counts().plot.bar()
    return capturas_df



def separar_imagenes():
    from shutil import copyfile
    capturas_df = get_file_exists()
    capturas_df = capturas_df.loc[capturas_df["file_exists_status"]  == True]
    init_categorias_folders()
    bar = progressbar.ProgressBar(
        maxval=capturas_df.shape[0],
        widgets=progressbar_settings,
    )
    bar.start()
    for index, row in capturas_df.iterrows():
        localizaciones_df = get_capturas_localizaciones_df()
        categoria = get_categoria(localizaciones_df, row["DescripcionEnEstudio"])
        categoria_path = os.path.join(capturas_root,"EstudiosSeparados",str(categoria))
        target_path = os.path.join(categoria_path,os.path.basename(row["url"]))
        source_path = os.path.join(capturas_root, row["url"])
        if index % 10 == 0:
            print (f"{source_path} to {target_path}")
        if os.path.isfile(source_path):
            copyfile(source_path, target_path)
        bar.update(index)

    bar.finish()
    # capturas_df["file_exists_status"].value_counts().plot.bar()
    return capturas_df


