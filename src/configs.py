from enum import Enum
import pandas as pd


class ModelConfigs(Enum):
    NUM_ITERS = 500
    SELECTION_THRESHOLD = 0.0
    PENALTIES = [10, 5, 2, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001, 0.00001]
    MAX_SELECTION = 100_000
    MIN_SELECTION = 10_000


class Languages(Enum):
    English = "en_core_web_sm"
    Italian = "it_core_news_sm"
    German = "de_core_news_sm"
    Spanish = "es_core_news_sm"
    Greek = "el_core_news_sm"
    Dutch = "nl_core_news_sm"
    Portuguese = "pt_core_news_sm"
    French = "fr_core_news_sm"
    Chinese = "zh_core_news_sm"
    Danish = "da_core_news_sm"
    Japanese = "ja_core_news_sm"
    Lithuanian = "lt_core_news_sm"
    Norvegian = "nb_core_news_sm"
    Polish = "pl_core_news_sm"
    Romanian = "ro_core_news_sm"
    Russian = "ru_core_news_sm"
    MultiLanguage = "xx_ent_wiki_sm"


class SupportedFiles(Enum):
    xlsx = (lambda x: pd.read_excel(x, dtype=str),)
    csv = (lambda x: pd.read_csv(x, dtype=str),)
    parquet = (lambda x: pd.read_parquet(x),)
