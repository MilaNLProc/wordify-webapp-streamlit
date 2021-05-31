from src.configs import Languages
from src.utils import read_file, download_button
from src.plotting import plot_labels_prop, plot_nchars, plot_score
from src.preprocessing import Lemmatizer, PreprocessingPipeline, encode
from src.wordifier import wordifier
import streamlit as st


def write(session, uploaded_file):

    if not uploaded_file:
        st.markdown(
            """
            Hi, welcome to __Wordify__! :rocket:

            Start by uploading a file - CSV, XLSX (avoid Strict Open XML Spreadsheet format [here](https://stackoverflow.com/questions/62800822/openpyxl-cannot-read-strict-open-xml-spreadsheet-format-userwarning-file-conta)),
            or PARQUET are currently supported.

            Once you have uploaded the file, __Wordify__ will show an interactive UI through which
            you'll be able to interactively decide the text preprocessing steps, their order, and
            proceed to Wordify your text.

            If you're ready, let's jump in:

            :point_left: upload a file via the upload widget in the sidebar!

            NOTE: whenever you want to reset everything, simply refresh the page.
            """
        )

    elif uploaded_file:

        # ==== 1. READ FILE ==== #
        with st.spinner("Reading file"):
            # TODO: write parser function that automatically understands format
            data = read_file(uploaded_file)

        # 2. CREATE UI TO SELECT COLUMNS
        col1, col2, col3 = st.beta_columns(3)
        with col1:
            language = st.selectbox("Select language", [i.name for i in Languages])
            with st.beta_expander("Description"):
                st.markdown(
                    f"Select a language amongst those supported: {', '.join([f'`{i.name}`' for i in Languages])}. This will be used to lemmatize and remove stopwords."
                )
        with col2:
            cols_options = [""] + data.columns.tolist()
            label_column = st.selectbox("Select label column name", cols_options, index=0)
            with st.beta_expander("Description"):
                st.markdown("Select the column containing the labels.")

            if label_column:
                plot = plot_labels_prop(data, label_column)
                if plot:
                    st.altair_chart(plot, use_container_width=True)

        with col3:
            text_column = st.selectbox("Select text column name", cols_options, index=0)
            with st.beta_expander("Description"):
                st.markdown("Select the column containing the texts.")

            if text_column:
                st.altair_chart(plot_nchars(data, text_column), use_container_width=True)

        # ==== 2.1 CREATE UI FOR ADVANCED OPTIONS ==== #
        with st.beta_expander("Advanced options"):

            steps_options = list(PreprocessingPipeline.pipeline_components().keys())

            # stopwords option and
            col1, col2 = st.beta_columns([1, 3])
            with col1:
                st.markdown("Remove stopwords (uses Spacy vocabulary)")
            with col2:
                remove_stopwords_elem = st.empty()

            # lemmatization option
            col1, col2 = st.beta_columns([1, 3])
            with col1:
                st.markdown("Lemmatizes text (uses Spacy)")
            with col2:
                lemmatization_elem = st.empty()

            # pre-lemmatization cleaning steps and
            # post-lemmatization cleaning steps
            col1, col2 = st.beta_columns([1, 3])
            with col1:
                st.markdown(
                    f"""
                    Define a pipeline of cleaning steps that is applied before and/or after lemmatization.
                    The available cleaning steps are:\n
                    {", ".join([f"`{x.replace('_', ' ').title()}`" for x in steps_options])}
                    """
                )
            with col2:
                pre_steps_elem = st.empty()
                post_steps_elem = st.empty()
                reset_button = st.empty()

            # implement reset logic
            if reset_button.button("Reset steps"):
                session.run_id += 1

            pre_steps = pre_steps_elem.multiselect(
                "Select pre-lemmatization preprocessing steps (ordered)",
                options=steps_options,
                default=steps_options[1:],
                format_func=lambda x: x.replace("_", " ").title(),
                key=session.run_id,
            )
            post_steps = post_steps_elem.multiselect(
                "Select post-lemmatization processing steps (ordered)",
                options=steps_options,
                default=steps_options[-4:],
                format_func=lambda x: x.replace("_", " ").title(),
                key=session.run_id,
            )
            remove_stopwords = remove_stopwords_elem.checkbox(
                "Remove stopwords",
                value=True,
                key=session.run_id,
            )
            lemmatization = lemmatization_elem.checkbox(
                "Lemmatize text",
                value=True,
                key=session.run_id,
            )

        # show sample checkbox
        col1, col2 = st.beta_columns([1, 2])
        with col1:
            show_sample = st.checkbox("Show sample of preprocessed text")

        # initialize text preprocessor
        preprocessing_pipeline = PreprocessingPipeline(
            pre_steps=pre_steps,
            lemmatizer=Lemmatizer(
                language=language,
                remove_stop=remove_stopwords,
                lemmatization=lemmatization,
            ),
            post_steps=post_steps,
        )

        # ==== 3. PROVIDE FEEDBACK ON OPTIONS ==== #
        if show_sample and not (label_column and text_column):
            st.warning("Please select `label` and `text` columns")

        elif show_sample and (label_column and text_column):
            sample_data = data.sample(5)
            sample_data[f"preprocessed_{text_column}"] = preprocessing_pipeline(
                sample_data[text_column]
            ).values
            st.table(sample_data.loc[:, [label_column, text_column, f"preprocessed_{text_column}"]])

        # ==== 4. RUN ==== #
        run_button = st.button("Wordify!")
        if run_button and not (label_column and text_column):
            st.warning("Please select `label` and `text` columns")

        elif run_button and (label_column and text_column) and not session.process:

            with st.spinner("Process started"):
                # data = data.head()
                data[f"preprocessed_{text_column}"] = preprocessing_pipeline(
                    data[text_column]
                ).values

                inputs = encode(data[f"preprocessed_{text_column}"], data[label_column])
                session.posdf, session.negdf = wordifier(**inputs)
            st.success("Wordified!")

            # session.posdf, session.negdf = process(data, text_column, label_column)
            session.process = True

        # ==== 5. RESULTS ==== #
        if session.process and (label_column and text_column):
            st.markdown("")
            st.markdown("")
            st.header("Results")

            # col1, col2, _ = st.beta_columns(3)
            col1, col2, col3 = st.beta_columns([2, 3, 3])

            with col1:
                label = st.selectbox("Select label", data[label_column].unique().tolist())
                # # with col2:
                # thres = st.slider(
                #     "Select threshold",
                #     min_value=0,
                #     max_value=100,
                #     step=1,
                #     format="%f",
                #     value=30,
                # )
                show_plots = st.checkbox("Show plots of top 100")

            with col2:
                st.subheader(f"Words __positively__ identifying label `{label}`")
                st.write(
                    session.posdf[session.posdf[label_column] == label].sort_values(
                        "score", ascending=False
                    )
                )
                download_button(session.posdf, "positive_data")
                if show_plots:
                    st.altair_chart(
                        plot_score(session.posdf, label_column, label),
                        use_container_width=True,
                    )

            with col3:
                st.subheader(f"Words __negatively__ identifying label `{label}`")
                st.write(
                    session.negdf[session.negdf[label_column] == label].sort_values(
                        "score", ascending=False
                    )
                )
                download_button(session.negdf, "negative_data")
                if show_plots:
                    st.altair_chart(
                        plot_score(session.negdf, label_column, label),
                        use_container_width=True,
                    )
