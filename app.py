import streamlit as st
from src.utils import get_logo
from src import session_state
from src.pages import (
    home,
    faq,
    about,
)
from src.configs import SupportedFiles

# app configs
st.set_page_config(
    page_title="Wordify",
    layout="wide",
    page_icon="./assets/logo.png",
)

# session state
session = session_state.get(process=False, run_id=0, posdf=None, negdf=None)


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

with st.sidebar.beta_container():
    st.sidebar.header("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

page = PAGES[selection]

# FILE UPLOADER
with st.sidebar.beta_container():
    st.markdown("")
    st.markdown("")
    st.header("Upload file")
    uploaded_file = st.sidebar.file_uploader("Select file", type=[i.name for i in SupportedFiles])


# FOOTER
with st.sidebar.beta_container():
    st.markdown("")
    st.markdown("")
    st.markdown(
        """
        <span style="font-size: 0.75em">Built with &hearts; by [`Pietro Lesci`](https://pietrolesci.github.io/) and [`MilaNLP`](https://twitter.com/MilaNLProc?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor)</span>
        """,
        unsafe_allow_html=True,
    )


# ==== MAIN ==== #
with st.beta_container():
    st.title("Wordify")
    st.markdown(
        """
        Wordify makes it easy to identify words that discriminate categories in textual data.

        Let's explain Wordify with an example. Imagine you are thinking about having a glass
        of wine :wine_glass: with your friends :man-man-girl-girl: and you have to buy a bottle.
        You know you like `bold`, `woody` wine but are unsure which one to choose.
        You wonder whether there are some words that describe each type of wine.
        Since you are a researcher :female-scientist: :male-scientist:, you decide to approach
        the problem scientifically :microscope:. That's where Wordify comes to the rescue!
        """
    )


page.write(session, uploaded_file)