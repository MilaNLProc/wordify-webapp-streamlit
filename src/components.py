import time

import pandas as pd
import streamlit as st

from src.configs import ColumnNames, Languages, PreprocessingConfigs, SupportedFiles
from src.preprocessing import PreprocessingPipeline
from src.utils import get_col_indices
from src.wordifier import input_transform, output_transform, wordifier


def docs():
    steps_options = list(PreprocessingPipeline.pipeline_components().keys())

    with st.expander("Documentation for the Advanced Options"):
        component_name = st.selectbox(
            "Select a processing step to see docs",
            options=[""] + steps_options,
            index=1,
            format_func=lambda x: x.replace("_", " ").title(),
            help="Select a processing step to see the relative documentation",
        )

        pipe_component = PreprocessingPipeline.pipeline_components().get(component_name)
        if pipe_component is not None:
            st.help(pipe_component)


def form(df):
    st.subheader("Parameters")
    with st.form("Wordify form"):
        col1, col2, col3 = st.columns(3)
        cols = [""] + df.columns.tolist()
        text_index, label_index = get_col_indices(cols)
        with col1:
            label_column = st.selectbox(
                "Select label column",
                cols,
                index=label_index,
                help="Select the column containing the labels",
            )
        with col2:
            text_column = st.selectbox(
                "Select text column",
                cols,
                index=text_index,
                help="Select the column containing the text",
            )
        with col3:
            language = st.selectbox(
                "Select language",
                [i.name for i in Languages],
                help="""
                    Select the language of your texts amongst the supported one. If we currently do
                    not support it, feel free to open an issue
                """,
            )

        with st.expander("Advanced Options"):
            disable_preprocessing = st.checkbox("Disable Preprocessing", False)

            if not disable_preprocessing:
                steps_options = list(PreprocessingPipeline.pipeline_components().keys())

                pre_steps = st.multiselect(
                    "Select pre-lemmatization processing steps (ordered)",
                    options=steps_options,
                    default=[
                        steps_options[i] for i in PreprocessingConfigs.DEFAULT_PRE.value
                    ],
                    format_func=lambda x: x.replace("_", " ").title(),
                    help="Select the processing steps to apply before the text is lemmatized",
                )

                lammatization_options = list(
                    PreprocessingPipeline.lemmatization_component().keys()
                )
                lemmatization_step = st.selectbox(
                    "Select lemmatization",
                    options=lammatization_options,
                    index=PreprocessingConfigs.DEFAULT_LEMMA.value,
                    help="Select lemmatization procedure. This is automatically disabled when the selected language is Chinese or MultiLanguage.",
                )

                post_steps = st.multiselect(
                    "Select post-lemmatization processing steps (ordered)",
                    options=steps_options,
                    default=[
                        steps_options[i]
                        for i in PreprocessingConfigs.DEFAULT_POST.value
                    ],
                    format_func=lambda x: x.replace("_", " ").title(),
                    help="Select the processing steps to apply after the text is lemmatized",
                )

        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
        if submitted:

            start_time = time.time()

            # warnings about inputs
            language_specific_warnings(
                pre_steps, post_steps, lemmatization_step, language
            )

            # preprocess
            if not disable_preprocessing:
                with st.spinner("Step 1/4: Preprocessing text"):
                    pipe = PreprocessingPipeline(
                        language, pre_steps, lemmatization_step, post_steps
                    )
                    df = pipe.vaex_process(df, text_column)
            else:
                with st.spinner(
                    "Step 1/4: Preprocessing has been disabled - doing nothing"
                ):
                    df = df.rename(
                        columns={text_column: ColumnNames.PROCESSED_TEXT.value}
                    )
                    time.sleep(1.2)

            # prepare input
            with st.spinner("Step 2/4: Preparing inputs"):
                input_dict = input_transform(
                    df[ColumnNames.PROCESSED_TEXT.value], df[label_column]
                )

            # wordify
            with st.spinner("Step 3/4: Wordifying"):
                pos, neg = wordifier(**input_dict)

            # prepare output
            with st.spinner("Step 4/4: Preparing outputs"):
                new_df = output_transform(pos, neg)

            end_time = time.time()
            meta_data = {
                "vocab_size": input_dict["X"].shape[1],
                "n_instances": input_dict["X"].shape[0],
                "vocabulary": pd.DataFrame({"Vocabulary": input_dict["X_names"]}),
                "labels": pd.DataFrame({"Labels": input_dict["y_names"]}),
                "time": round(end_time - start_time),
            }

            return new_df, meta_data


def faq():
    st.subheader("Frequently Asked Questions")
    with st.expander("What is Wordify?"):
        st.markdown(
            """
            __Wordify__ is a way to find out which n-grams (i.e., words and concatenations of words) are most indicative for each of your dependent
            variable values.
            """
        )

    with st.expander("What happens to my data?"):
        st.markdown(
            """
            Nothing. We never store the data you upload on disk: it is only kept in memory for the
            duration of the modeling, and then deleted. We do not retain any copies or traces of
            your data.
            """
        )

    with st.expander("What input formats do you support?"):
        st.markdown(
            f"""
            We currently support {", ".join([i.name for i in SupportedFiles])}.
            """
        )

    with st.expander("Do I need to preprocess my data?"):
        st.markdown(
            """
            No, there is no need to preprocess your text, we will take of it. 
            However, if you wish to do so, turn off preprocessing in the `Advanced
            Settings` in the interactive UI.
        """
        )

    with st.expander("What languages are supported?"):
        st.markdown(
            f"""
            Currently we support: {", ".join([i.name for i in Languages])}.
            """
        )

    with st.expander("How does it work?"):
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

    with st.expander("What libraries do you use?"):
        st.markdown(
            """
            We leverage the power of many great libraries in the Python ecosystem:
            - `Streamlit`
            - `Pandas`
            - `Numpy`
            - `Spacy`
            - `Scikit-learn`
            - `Vaex`
            """
        )

    with st.expander("How much data do I need?"):
        st.markdown(
            """
            We recommend at least 2000 instances, the more, the better. With fewer instances, the
            results are less replicable and reliable.
            """
        )

    with st.expander("Is there a paper I can cite?"):
        st.markdown(
            """
            Yes, please! Cite [Wordify: A Tool for Discovering and Differentiating Consumer Vocabularies](https://academic.oup.com/jcr/article/48/3/394/6199426)
            ```
            @article{10.1093/jcr/ucab018,
                author = {Hovy, Dirk and Melumad, Shiri and Inman, J Jeffrey},
                title = "{Wordify: A Tool for Discovering and Differentiating Consumer Vocabularies}",
                journal = {Journal of Consumer Research},
                volume = {48},
                number = {3},
                pages = {394-414},
                year = {2021},
                month = {03},
                abstract = "{This work describes and illustrates a free and easy-to-use online text-analysis tool for understanding how consumer word use varies across contexts. The tool, Wordify, uses randomized logistic regression (RLR) to identify the words that best discriminate texts drawn from different pre-classified corpora, such as posts written by men versus women, or texts containing mostly negative versus positive valence. We present illustrative examples to show how the tool can be used for such diverse purposes as (1) uncovering the distinctive vocabularies that consumers use when writing reviews on smartphones versus PCs, (2) discovering how the words used in Tweets differ between presumed supporters and opponents of a controversial ad, and (3) expanding the dictionaries of dictionary-based sentiment-measurement tools. We show empirically that Wordify’s RLR algorithm performs better at discriminating vocabularies than support vector machines and chi-square selectors, while offering significant advantages in computing time. A discussion is also provided on the use of Wordify in conjunction with other text-analysis tools, such as probabilistic topic modeling and sentiment analysis, to gain more profound knowledge of the role of language in consumer behavior.}",
                issn = {0093-5301},
                doi = {10.1093/jcr/ucab018},
                url = {https://doi.org/10.1093/jcr/ucab018},
                eprint = {https://academic.oup.com/jcr/article-pdf/48/3/394/40853499/ucab018.pdf},
            }
            ```
            """
        )

    with st.expander("How can I reach out to the Wordify team?"):
        st.markdown(contacts(), unsafe_allow_html=True)


def presentation():
    st.markdown(
        """
        Wordify makes it easy to identify words that discriminate categories in textual data.
        
        :point_left: Start by uploading a file. *Once you upload the file, __Wordify__ will
        show an interactive UI*.
        """
    )

    st.subheader("Quickstart")
    st.markdown(
        """
        - There is no need to preprocess your text, we will take care of it. However, if you wish to
        do so, turn off preprocessing in the `Advanced Settings` in the interactive UI.

        - We expect a file with two columns: `label` with the labels and `text` with the texts (the names are case insensitive). If
        you provide a file following this naming convention, Wordify will automatically select the
        correct columns. However, if you wish to use a different nomenclature, you will be asked to
        provide the column names in the interactive UI.

        - Maintain a stable connection with the Wordify page until you download the data. If you refresh the page, 
        a new Wordify session is created and your progress is lost.
        
        - Wordify performances depend on the length of the individual texts in your file. The longer the texts, the higher
        the chance that Wordify considers many n-grams. More n-grams means more data to analyse in each run.
        We tailored Wordify performance for files of approximately 5'000 lines or 50k n-grams. In such cases we expect a runtime
        between 90 seconds and 10 minutes. If your file is big, try to apply a stricter preprocessing of the text in the `Advanced Options` section.
        If this is not enough, please do feel free to reach out to us directly so we can help.
        """
    )

    st.subheader("Input format")
    st.markdown(
        """
        Please note that your file must have a column with the texts and a column with the labels,
        for example
        """
    )
    st.table(
        {
            "text": ["A review", "Another review", "Yet another one", "etc"],
            "label": ["Good", "Bad", "Good", "etc"],
        }
    )

    st.subheader("Output format")
    st.markdown(
        """
        As a result of the process, you will get a file containing 4 columns:
        - `Word`: the n-gram (i.e., a word or a concatenation of words) considered
        - `Score`: the wordify score, between 0 and 1, of how important is `Word` to discrimitate `Label`
        - `Label`: the label that `Word` is discriminating
        - `Correlation`: how `Word` is correlated with `Label` (e.g., "negative" means that if `Word` is present in the text then the label is less likely to be `Label`)

        for example
        """
    )

    st.table(
        {
            "Word": ["good", "awful", "bad service", "etc"],
            "Score": ["0.52", "0.49", "0.35", "etc"],
            "Label": ["Good", "Bad", "Good", "etc"],
            "Correlation": ["positive", "positive", "negative", "etc"],
        }
    )


def footer():
    st.sidebar.markdown(
        """
        <span style="font-size: 0.75em">Built with &hearts; by [`Pietro Lesci`](https://pietrolesci.github.io/) and [`MilaNLP`](https://twitter.com/MilaNLProc?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor).</span>
        """,
        unsafe_allow_html=True,
    )


def contacts():
    return """
    You can reach out to us via email, phone, or via mail
    
    - :email: wordify@unibocconi.it
    
    - :telephone_receiver: +39 02 5836 2604
    
    - :postbox: Via Röntgen n. 1, Milan 20136 (ITALY)


    <iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d2798.949796165441!2d9.185730115812493!3d45.450667779100726!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x4786c405ae6543c9%3A0xf2bb2313b36af88c!2sVia%20Guglielmo%20R%C3%B6ntgen%2C%201%2C%2020136%20Milano%20MI!5e0!3m2!1sit!2sit!4v1569325279433!5m2!1sit!2sit" frameborder="0" style="border:0; width: 100%; height: 312px;" allowfullscreen></iframe>
    """


def analysis(outputs):

    df, meta_data = outputs

    st.subheader("Results")
    st.markdown(
        """
    Wordify successfully run and you can now look at the results before downloading the wordified file.
    In particular, you can use the slider to filter only those words that have a `Score` above (>=) a certain threshold.
    For meaningful results, we suggest keeping the threshold to 0.25.
    """
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        threshold = st.slider(
            "Select threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            value=0.25,
            help="To return everything, select 0.",
        )
        subset_df = df.loc[df["Score"] >= threshold].reset_index(drop=True)
        st.write(subset_df)

    with col2:
        st.markdown("**Some info about your data**")
        st.markdown(
            f"""
            Your input file contained {meta_data["n_instances"]:,} rows and 
            Wordify took {meta_data["time"]:,} seconds to run.
            
            The total number of n-grams Wordify considered is {meta_data["vocab_size"]:,}.
            With the current selected threshold on the `Score` (>={threshold}) the output contains {subset_df["Word"].nunique():,}
            unique n-grams.
            """
        )

        with st.expander("Vocabulary"):
            st.markdown(
                "The table below shows all candidate n-grams that Wordify considered"
            )
            st.write(meta_data["vocabulary"])

        with st.expander("Labels"):
            st.markdown(
                "The table below summarizes the labels that your file contained"
            )
            st.write(meta_data["labels"])

    return subset_df


# warning for Chinese and MultiLanguage
def language_specific_warnings(pre_steps, post_steps, lemmatization_step, language):

    if language in ("MultiLanguage", "Chinese") and (
        "remove_non_words" in pre_steps or "remove_non_words" in post_steps
    ):
        msg = """
        NOTE: for Chinese and MultiLanguage we automatically substitute `remove_non_words` with
        `remove_numbers` and `remove_punctuation` to avoid wrong results.
        """
        st.info(msg)

    msg = "NOTE: for Chinese and MultiLanguage we turn-off lemmatization automatically."
    if lemmatization_step == "Spacy lemmatizer (keep stopwords)" and language in (
        "MultiLanguage",
        "Chinese",
    ):
        st.info(msg)

    elif lemmatization_step == "Spacy lemmatizer (remove stopwords)" and language in (
        "MultiLanguage",
        "Chinese",
    ):
        st.info(
            msg
            + " However we will still remove stopwords since you selected `Spacy lemmatizer (remove stopwords)`."
        )
