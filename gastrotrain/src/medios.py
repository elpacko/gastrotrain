import os
from csv import reader

import pandas as pd
import progressbar
from settings import get_settings

app_settings = get_settings()

def __init__():
    pass


def get_df():
    df = pd.read_csv(app_settings["default"]["DATASOURCE"])
    return df



def get_file_exists():
    df = get_df()
    file_exists_status = []
    bar = progressbar.ProgressBar(
        maxval=df.shape[0],
        widgets=[progressbar.Bar("=", "[", "]"), " ", progressbar.Percentage()],
    )
    bar.start()
    capturas_root = app_settings["default"]["CAPTURAS_ROOT"]
    for index, row in df.iterrows():
        file_name = capturas_root + row["url"]
        file_exists = os.path.isfile(file_name)
        bar.update(index)
        file_exists_status.append(file_exists)
    df["file_exists_status"] = file_exists_status
    bar.finish()
    df.to_csv(
        "file_exists_status.csv", index=False, encoding="utf-8"
    )  # False: not include index
    df["file_exists_status"].value_counts().plot.bar()
    return df
