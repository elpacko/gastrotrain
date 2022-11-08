import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
from medios import get_df
from nltk.tokenize import word_tokenize

df = get_df()

def tokenize(descripcion):
    return word_tokenize(descripcion)

