import streamlit as st
from src.utils import get_logo, read_file, convert_df
from src.components import form, faq, presentation, footer, about


# app configs
st.set_page_config(
    page_title="Wordify",
    initial_sidebar_state="expanded",
    layout="centered",
    page_icon="./assets/logo.png",
    menu_items={
        'Get Help': "https://github.com/MilaNLProc/wordify-webapp-streamlit/issues/new",
        'Report a Bug': "https://github.com/MilaNLProc/wordify-webapp-streamlit/issues/new",
        'About': about(),
    }
)

# logo
st.sidebar.image(get_logo("./assets/logo.png"))

# title
st.title("Wordify")

# file uploader
uploaded_fl = st.sidebar.file_uploader(
    label="Choose a file",
    type=["csv", "parquet", "tsv", "xlsx"],
    accept_multiple_files=False,
    help="""
        Supported formats:
        - CSV
        - TSV 
        - PARQUET
        - XLSX (do not support [Strict Open XML Spreadsheet format](https://stackoverflow.com/questions/62800822/openpyxl-cannot-read-strict-open-xml-spreadsheet-format-userwarning-file-conta))
    """,
)

if not uploaded_fl:
    presentation()
    faq()
else:
    df = read_file(uploaded_fl)
    new_df = form(df)
    if new_df is not None:
        payload = convert_df(new_df)
        st.download_button(
            label="Download data as CSV",
            data=payload,
            file_name="wordify_results.csv",
            mime="text/csv",
        )


# footer
footer()
