import os
import re

import nltk
import pandas as pd
import unidecode
# from gensim.parsing.preprocessing import remove_stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.spatial import distance
from settings import get_settings

app_settings = get_settings()
categorias_path = os.path.join(app_settings["default"]["CAPTURAS_ROOT"], app_settings["default"]["LOCALIZACIONES"])
def jaccard_similarity(A, B):
    #Find intersection of two sets
    nominator = A.intersection(B)
    #Find union of two sets
    denominator = A.union(B)
    #Take the ratio of sizes
    similarity = len(nominator)/len(denominator)
    return similarity

def tokenize(descripcion):
    return word_tokenize(descripcion)


def stemSentence(sentence):
    token_words = word_tokenize(sentence)
    stem_sentence = [
        unidecode.unidecode(re.sub(r"\W+", "", word)) for word in token_words
    ]
    return " ".join(stem_sentence)


def extraer_categorias(df):
    descripciones_unicas = df.DescripcionEnEstudio.unique()
    descripciones_unicas = [stemSentence(x) for x in descripciones_unicas]
    vectorizer_cv = CountVectorizer(analyzer="word")
    X_cv = vectorizer_cv.fit_transform(descripciones_unicas)


def get_categorias(source_df):
    if os.path.exists(categorias_path):
        df = pd.read_csv(categorias_path)
        return df
    else:
        return extraer_categorias(source_df)
