import streamlit as st
from src.configs import Languages


def write(*args):

    # ==== HOW IT WORKS ==== #
    with st.beta_container():
        st.markdown("")
        st.markdown("")
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
        st.markdown("")
        st.markdown("")
        st.header("Steps")
        st.subheader("Step 1 - Prepare your data")
        st.markdown(
            """
            Create an Excel or CSV file with two columns for each row:

            - a column with the name or the label identifying a specific object or class (e.g., in our
            wine example above it would be the type of wine or the name of a specific brand). It is
            common practice naming this column `label`

            - a column with the text describing that specific object or class (e.g., in the wine example
            above it could be the description that you find on the rear of the bottle label). It is
            common practice naming this column `text`

            To have reliable results, we suggest providing at least 2000 labelled texts. If you provide
            less we will still wordify your file, but the results should then be taken with a grain of
            salt.

            Consider that we also support multi-language texts, therefore you'll be able to
            automatically discriminate between international wines, even if your preferred Italian
            producer does not provide you with a description written in English!
            """
        )

        st.subheader("Step 2 - Upload your file and Wordify!")
        st.markdown(
            """
            Once you have prepared your Excel or CSV file, click the "Browse File" button.
            Browse for your file.
            Choose the language of your texts (select multi-language if your file contains text in
            different languages).
            Push the "Wordify|" button, set back, and wait for wordify to do its tricks.

            Depending on the size of your data, the process can take from 1 minute to 5 minutes
            """
        )

    # ==== FAQ ==== #
    with st.beta_container():
        st.markdown("")
        st.markdown("")
        st.header(":question:Frequently Asked Questions")
        with st.beta_expander("What is Wordify?"):
            st.markdown(
                """
                Wordify is a way to find out which terms are most indicative for each of your dependent
                variable values.
                """
            )

        with st.beta_expander("What happens to my data?"):
            st.markdown(
                """
                Nothing. We never store the data you upload on disk: it is only kept in memory for the
                duration of the modeling, and then deleted. We do not retain any copies or traces of
                your data.
                """
            )

        with st.beta_expander("What input formats do you support?"):
            st.markdown(
                """
                The file you upload should be .xlsx, with two columns: the first should be labeled
                'text' and contain all your documents (e.g., tweets, reviews, patents, etc.), one per
                line. The second column should be labeled 'label', and contain the dependent variable
                label associated with each text (e.g., rating, author gender, company, etc.).
                """
            )

        with st.beta_expander("How does it work?"):
            st.markdown(
                """
                It uses a variant of the Stability Selection algorithm
                [(Meinshausen and Bühlmann, 2010)](https://rss.onlinelibrary.wiley.com/doi/full/10.1111/j.1467-9868.2010.00740.x)
                to fit hundreds of logistic regression models on random subsets of the data, using
                different L1 penalties to drive as many of the term coefficients to 0. Any terms that
                receive a non-zero coefficient in at least 30% of all model runs can be seen as stable
                indicators.
                """
            )

        with st.beta_expander("How much data do I need?"):
            st.markdown(
                """
                We recommend at least 2000 instances, the more, the better. With fewer instances, the
                results are less replicable and reliable.
                """
            )

        with st.beta_expander("Is there a paper I can cite?"):
            st.markdown(
                """
                Yes please! Reference coming soon...
                """
            )

        with st.beta_expander("What languages are supported?"):
            st.markdown(
                f"""
                Currently we support: {", ".join([i.name for i in Languages])}.
                """
            )
