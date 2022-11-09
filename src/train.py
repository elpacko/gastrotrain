import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from categorias import get_categorias
from medios import get_df

df = get_df()
categorias = get_categorias(df)
