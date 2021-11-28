import streamlit as st

from src import session_state
from src.configs import SupportedFiles
from src.pages import about, faq, home
from src.utils import get_logo

# app configs
st.set_page_config(
    page_title="Wordify",
    layout="wide",
    page_icon="./assets/logo.png",
)

# session state
session = session_state.get(
    process=False, run_id=0, posdf=None, negdf=None, uploaded_file_id=0
)


# ==== SIDEBAR ==== #
# LOGO
client_logo = get_logo("./assets/logo.png")
with st.sidebar.beta_container():
    st.image(client_logo)

# NAVIGATION
PAGES = {
    "Home": home,
    "FAQ": faq,
    "About": about,
}

st.sidebar.header("Navigation")
# with st.sidebar.beta_container():
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]

# FILE UPLOADER
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.header("Upload file")
# with st.sidebar.beta_container():
uploaded_file = st.sidebar.file_uploader(
    "Select file", type=[i.name for i in SupportedFiles]
)


# FOOTER
# with st.sidebar.beta_container():
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown(
    """
    <span style="font-size: 0.75em">Built with &hearts; by [`Pietro Lesci`](https://pietrolesci.github.io/) and [`MilaNLP`](https://twitter.com/MilaNLProc?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor)</span>
    """,
    unsafe_allow_html=True,
)
st.sidebar.info(
    "Something not working? Consider [filing an issue](https://github.com/MilaNLProc/wordify-webapp-streamlit/issues/new)"
)


# ==== MAIN ==== #
with st.beta_container():
    st.title("Wordify")


page.write(session, uploaded_file)
