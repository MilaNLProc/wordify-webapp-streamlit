import base64

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image

from .configs import SupportedFiles


@st.cache
def get_logo(path):
    return Image.open(path)


@st.experimental_memo
def read_file(uploaded_file) -> pd.DataFrame:
    file_type = uploaded_file.name.split(".")[-1]
    read_fn = SupportedFiles[file_type].value[0]
    df = read_fn(uploaded_file)
    df = df.dropna()
    return df


@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False, sep=";").encode("utf-8")


def download_button(dataframe: pd.DataFrame, name: str):
    csv = dataframe.to_csv(index=False)
    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{name}.csv">Download</a>'
    st.write(href, unsafe_allow_html=True)


def plot_labels_prop(data: pd.DataFrame, label_column: str):

    unique_value_limit = 100

    if data[label_column].nunique() > unique_value_limit:

        st.warning(
            f"""
        The column you selected has more than {unique_value_limit}.
        Are you sure it's the right column? If it is, please note that
        this will impact __Wordify__ performance.
        """
        )

        return

    source = (
        data[label_column]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "Labels", label_column: "Counts"})
    )
    source["Props"] = source["Counts"] / source["Counts"].sum()
    source["Proportions"] = (source["Props"].round(3) * 100).map("{:,.2f}".format) + "%"

    bars = (
        alt.Chart(source)
        .mark_bar()
        .encode(
            x=alt.X("Labels:O", sort="-y"),
            y="Counts:Q",
        )
    )

    text = bars.mark_text(align="center", baseline="middle", dy=15).encode(
        text="Proportions:O"
    )

    return (bars + text).properties(height=300)


def plot_nchars(data: pd.DataFrame, text_column: str):
    source = data[text_column].str.len().to_frame()

    plot = (
        alt.Chart(source)
        .mark_bar()
        .encode(
            alt.X(
                f"{text_column}:Q", bin=True, axis=alt.Axis(title="# chars per text")
            ),
            alt.Y("count()", axis=alt.Axis(title="")),
        )
    )

    return plot.properties(height=300)


def plot_score(data: pd.DataFrame, label_col: str, label: str):

    source = (
        data.loc[data[label_col] == label]
        .sort_values("score", ascending=False)
        .head(100)
    )

    plot = (
        alt.Chart(source)
        .mark_bar()
        .encode(
            y=alt.Y("word:O", sort="-x"),
            x="score:Q",
        )
    )

    return plot.properties(height=max(30 * source.shape[0], 50))
