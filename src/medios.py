import os
import random
import re
from csv import reader
from shutil import copyfile, move
import pandas as pd
import progressbar
from nltk import edit_distance
from math import floor
from settings import get_settings

app_settings = get_settings()
capturas_root = app_settings["default"]["CAPTURAS_ROOT"]
capturas_datasource = os.path.join(capturas_root, app_settings["default"]["DATASOURCE"])
capturas_localizaciones = os.path.join(
    capturas_root, app_settings["default"]["LOCALIZACIONES"]
)
progressbar_settings = [progressbar.Bar("=", "[", "]"), " ", progressbar.Percentage()]
file_exists_path = os.path.join(capturas_root, "file_exists_status.csv")
full_categorias_subfolder = "EstudiosSeparados"
polipos_categorias_subfolder = "EstudiosPolipos"

def __init__():
    pass


def get_capturas_df():
    return pd.read_csv(capturas_datasource)


def get_capturas_localizaciones_df():
    return pd.read_csv(capturas_localizaciones)


def get_medios_path():
    return os.path.join(capturas_root, f"{full_categorias_subfolder}/")

def get_medios_polipos_path():
    return os.path.join(capturas_root, f"{polipos_categorias_subfolder}/")

def init_categorias_folders(target_folder):
    print("Creando folders de categorias")
    localizaciones_df = get_capturas_localizaciones_df()
    for _, localizacion_categoria in localizaciones_df.iterrows():
        categoria_path = os.path.join(
            capturas_root, target_folder, str(localizacion_categoria["Categoria"])
        )
        crear_directorio(categoria_path)


def init_categorias_polipos_folders(target_folder):
    print("Creando folders de categorias")
    for categoria in ["POLIPO", "NOPOLIPO"]:
        categoria_path = os.path.join(capturas_root, target_folder, categoria)
        crear_directorio(categoria_path)


def get_categoria(localizaciones_df, localizacion):
    localizacion = re.sub(r"\W+", "", localizacion)
    exact_match = localizaciones_df.loc[localizaciones_df["Nombre"] == localizacion]
    if exact_match.size == 1:
        return exact_match.item()
    match_dict = {}
    for _, localizacion_categoria in localizaciones_df.iterrows():
        match_dict[localizacion_categoria["Categoria"]] = edit_distance(
            localizacion_categoria["Nombre"].upper(), localizacion.upper()
        )
    min_key = min(match_dict, key=match_dict.get)
    return min_key


def get_tiene_polipo(localizacion):
    found_polipo = "NO" if not re.search("(?i)p.?lipo.*", localizacion) else ""
    return f"{found_polipo}POLIPO"


def crear_directorio(path):
    if not os.path.exists(path):
        os.mkdir(path)

def progressbar_common(size):
    return  progressbar.ProgressBar(
        maxval=size,
        widgets=progressbar_settings,
    )

def get_file_exists():
    print("Identificando imagenes existentes")
    if os.path.isfile(file_exists_path):
        return pd.read_csv(file_exists_path)
    capturas_df = get_capturas_df()
    file_exists_status = []
    bar = progressbar_common(capturas_df.shape[0])
    bar.start()
    for index, row in capturas_df.iterrows():
        file_name = capturas_root + row["url"]
        file_exists = os.path.isfile(file_name)
        bar.update(index)
        file_exists_status.append(file_exists)
    capturas_df["file_exists_status"] = file_exists_status
    bar.finish()
    capturas_df.to_csv(
        file_exists_path, index=False, encoding="utf-8"
    )  # False: not include index
    capturas_df["file_exists_status"].value_counts().plot.bar()
    return capturas_df


def separar_imagenes_localizaciones():
    from shutil import copyfile

    capturas_df = get_file_exists()
    capturas_df = capturas_df.loc[capturas_df["file_exists_status"] == True]
    init_categorias_folders(full_categorias_subfolder)
    bar = progressbar_common(capturas_df.shape[0])
    bar.start()
    for index, row in capturas_df.iterrows():
        localizaciones_df = get_capturas_localizaciones_df()
        categoria = get_categoria(localizaciones_df, row["DescripcionEnEstudio"])
        categoria_path = os.path.join(capturas_root, full_categorias_subfolder, str(categoria))
        target_path = os.path.join(categoria_path, os.path.basename(row["url"]))
        source_path = os.path.join(capturas_root, row["url"])
        if index % 10 == 0:
            print(f"{source_path} to {target_path}")
        if os.path.isfile(source_path):
            copyfile(source_path, target_path)
        bar.update(index)

    bar.finish()
    # capturas_df["file_exists_status"].value_counts().plot.bar()
    return capturas_df


def separar_imagenes_polipo_binario():
    capturas_df = get_file_exists()
    capturas_df = capturas_df.loc[capturas_df["file_exists_status"] == True]
    init_categorias_polipos_folders(polipos_categorias_subfolder)
    bar = progressbar_common(capturas_df.shape[0])
    bar.start()
    for index, row in capturas_df.iterrows():
        categoria = get_tiene_polipo(row["DescripcionEnEstudio"])
        categoria_path = os.path.join(capturas_root, polipos_categorias_subfolder, str(categoria))
        target_path = os.path.join(categoria_path, os.path.basename(row["url"]))
        source_path = os.path.join(capturas_root, row["url"])
        if index % 10 == 0:
            print(f"{source_path} to {target_path}")
        if os.path.isfile(source_path):
            copyfile(source_path, target_path)
        bar.update(index)

    bar.finish()
    # capturas_df["file_exists_status"].value_counts().plot.bar()
    return capturas_df

def select_sample_images():
    target_path = os.path.join(capturas_root, polipos_categorias_subfolder, "NOPOLIPO")
    source_path = os.path.join(capturas_root, polipos_categorias_subfolder, "NOPOLIPO_FULL")
    sample_path = os.path.join(capturas_root, polipos_categorias_subfolder, "POLIPO")
    files = os.listdir(source_path)
    sample_size = len(os.listdir(sample_path))
    sample = random.sample(files, sample_size) 
    for file in sample:
        copyfile(os.path.join(source_path, file), os.path.join(target_path, file))

def separar_train_test():
    categorias = ["POLIPO", "NOPOLIPO"]
    for categoria in categorias:
        
        source_path = os.path.join(capturas_root, polipos_categorias_subfolder,"train", categoria)
        sample_size = floor(len(os.listdir(source_path)) * .2)
        target_path = os.path.join(capturas_root, polipos_categorias_subfolder,"test", categoria)
        
        files = os.listdir(source_path)
        
        sample = random.sample(files, sample_size)
        for file in sample:
            move(os.path.join(source_path, file), os.path.join(target_path, file))