from src.configs import Languages
from src.utils import (
    encode,
    wordifier,
    download_button,
    TextPreprocessor,
    plot_labels_prop,
    plot_nchars,
    plot_score,
    get_logo,
    read_file,
)
import streamlit as st


def write(session, uploaded_file):

    if uploaded_file:

        # 1. READ FILE
        with st.spinner("Reading file"):
            # TODO: write parser function that automatically understands format
            data = read_file(uploaded_file)

        # 2. CREATE UI TO SELECT COLUMNS
        st.markdown("")
        st.markdown("")
        st.header("Process")

        col1, col2, col3 = st.beta_columns(3)
        with col1:
            language = st.selectbox("Select language", [i.name for i in Languages])
            with st.beta_expander("Description"):
                st.markdown(
                    f"Select a language of text amongst those supported: {', '.join([f'`{i.name}`' for i in Languages])}"
                )
        with col2:
            cols_options = [""] + data.columns.tolist()
            label_column = st.selectbox("Select label column name", cols_options, index=0)
            with st.beta_expander("Description"):
                st.markdown("Select the column containing the label")

            if label_column:
                st.altair_chart(plot_labels_prop(data, label_column), use_container_width=True)

        with col3:
            text_column = st.selectbox("Select text column name", cols_options, index=0)
            with st.beta_expander("Description"):
                st.markdown("Select the column containing the text")

            if text_column:
                st.altair_chart(plot_nchars(data, text_column), use_container_width=True)

        with st.beta_expander("Advanced options"):
            # Lemmatization option
            col1, col2 = st.beta_columns([1, 3])
            with col1:
                lemmatization_when_elem = st.empty()
            with col2:
                st.markdown("Choose lemmatization option")

            # stopwords option
            col1, col2 = st.beta_columns([1, 3])
            with col1:
                remove_stopwords_elem = st.empty()
            with col2:
                st.markdown("Choose stopword option")

            # cleaning steps
            col1, col2 = st.beta_columns([1, 3])
            with col1:
                cleaning_steps_elem = st.empty()
                reset_button = st.empty()
            with col2:
                st.markdown("Choose cleaning steps")

            # implement reset logic
            if reset_button.button("Reset steps"):
                session.run_id += 1

            steps_options = list(TextPreprocessor._cleaning_options().keys())
            cleaning_steps = cleaning_steps_elem.multiselect(
                "Select text processing steps (ordered)",
                options=steps_options,
                default=steps_options,
                format_func=lambda x: x.replace("_", " ").title(),
                key=session.run_id,
            )
            lemmatization_options = list(TextPreprocessor._lemmatization_options().keys())
            lemmatization_when = lemmatization_when_elem.selectbox(
                "Select when lemmatization happens",
                options=lemmatization_options,
                index=0,
                key=session.run_id,
            )
            remove_stopwords = remove_stopwords_elem.checkbox("Remove stopwords", value=True, key=session.run_id)

        # Show sample checkbox
        col1, col2 = st.beta_columns([1, 2])
        with col1:
            show_sample = st.checkbox("Show sample of preprocessed text")

        # initialize text preprocessor
        preprocessor = TextPreprocessor(
            language=language,
            cleaning_steps=cleaning_steps,
            lemmatizer_when=lemmatization_when,
            remove_stop=remove_stopwords,
        )

        # 3. PROVIDE FEEDBACK ON OPTIONS
        if show_sample and not (label_column and text_column):
            st.warning("Please select `label` and `text` columns")

        elif show_sample and (label_column and text_column):
            sample_data = data.sample(10)
            sample_data[f"preprocessed_{text_column}"] = preprocessor.fit_transform(sample_data[text_column]).values
            st.table(sample_data.loc[:, [label_column, text_column, f"preprocessed_{text_column}"]])

        # 4. RUN
        run_button = st.button("Wordify!")
        if run_button and not (label_column and text_column):
            st.warning("Please select `label` and `text` columns")

        elif run_button and (label_column and text_column) and not session.process:
            # data = data.head()
            data[f"preprocessed_{text_column}"] = preprocessor.fit_transform(data[text_column]).values

            inputs = encode(data[f"preprocessed_{text_column}"], data[label_column])
            session.posdf, session.negdf = wordifier(**inputs)
            st.success("Wordified!")

            # session.posdf, session.negdf = process(data, text_column, label_column)
            session.process = True

        # 5. RESULTS
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
                st.write(session.posdf[session.posdf[label_column] == label].sort_values("score", ascending=False))
                download_button(session.posdf, "positive_data")
                if show_plots:
                    st.altair_chart(plot_score(session.posdf, label_column, label), use_container_width=True)

            with col3:
                st.subheader(f"Words __negatively__ identifying label `{label}`")
                st.write(session.negdf[session.negdf[label_column] == label].sort_values("score", ascending=False))
                download_button(session.negdf, "negative_data")
                if show_plots:
                    st.altair_chart(plot_score(session.negdf, label_column, label), use_container_width=True)
