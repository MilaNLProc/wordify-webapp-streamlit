import base64
from typing import List, Tuple

import streamlit as st
from pandas.core.frame import DataFrame
from PIL import Image

from .configs import ColumnNames, SupportedFiles


def get_col_indices(cols: List) -> Tuple[int, int]:
    """Ugly but works"""
    cols = [i.lower() for i in cols]
    try:
        label_index = cols.index(ColumnNames.LABEL.value)
    except:
        label_index = 0

    try:
        text_index = cols.index(ColumnNames.TEXT.value)
    except:
        text_index = 0

    return text_index, label_index


@st.cache
def get_logo(path: str) -> Image:
    return Image.open(path)


@st.experimental_memo
def read_file(uploaded_file) -> DataFrame:
    file_type = uploaded_file.name.split(".")[-1]
    read_fn = SupportedFiles[file_type].value[0]
    df = read_fn(uploaded_file)
    df = df.dropna()
    return df


@st.cache
def convert_df(df: DataFrame) -> bytes:
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False, sep=";").encode("utf-8")


def download_button(dataframe: DataFrame, name: str) -> None:
    csv = dataframe.to_csv(index=False)
    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{name}.csv">Download</a>'
    st.write(href, unsafe_allow_html=True)
