import streamlit as st

# app configs
st.set_page_config(
    page_title="Wordify",
    initial_sidebar_state="expanded",
    layout="centered",
    page_icon="./assets/logo.png",
    menu_items={
        "Get Help": "https://github.com/MilaNLProc/wordify-webapp-streamlit/issues/new",
        "Report a Bug": "https://github.com/MilaNLProc/wordify-webapp-streamlit/issues/new",
        "About": "By the __Wordify__ team.",
    },  # type: ignore
)

# HACK: other streamlit complains that `set_page_config` is not the first command
if True:
    from src.components import analysis, docs, faq, footer, form, presentation
    from src.configs import SupportedFiles
    from src.utils import convert_df, get_logo, read_file

# logo
st.sidebar.image(get_logo("./assets/logo.png"))

# title
st.title("Wordify")

# file uploader
uploaded_fl = st.sidebar.file_uploader(
    label="Choose a file",
    type=[i.name for i in SupportedFiles],
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
    outputs = form(df)
    docs()

    # change or create session state
    if outputs is not None or "outputs" not in st.session_state:
        st.session_state["outputs"] = outputs

    # when procedure is performed
    if st.session_state["outputs"] is not None:

        df = analysis(st.session_state["outputs"])

        payload = convert_df(df)
        st.download_button(
            label="Download data as CSV",
            data=payload,
            file_name="wordify_results.csv",
            mime="text/csv",
        )

# footer
footer()
