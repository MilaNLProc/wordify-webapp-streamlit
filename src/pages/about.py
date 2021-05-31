import streamlit as st


def write(*args):
    # ==== Contacts ==== #
    with st.beta_container():
        st.markdown("")
        st.markdown("")
        st.header(":rocket:About us")

        st.markdown(
            """
            You can reach out to us via email, phone, or - if you are old-fashioned - via mail
            """
        )
        with st.beta_expander("Contacts"):

            _, col2 = st.beta_columns([0.5, 3])
            col2.markdown(
                """
                :email: wordify@unibocconi.it

                :telephone_receiver: +39 02 5836 2604

                :postbox: Via Röntgen n. 1, Milan 20136 (ITALY)
                """
            )

        st.write(
            """
            <iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d2798.949796165441!2d9.185730115812493!3d45.450667779100726!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x4786c405ae6543c9%3A0xf2bb2313b36af88c!2sVia%20Guglielmo%20R%C3%B6ntgen%2C%201%2C%2020136%20Milano%20MI!5e0!3m2!1sit!2sit!4v1569325279433!5m2!1sit!2sit" frameborder="0" style="border:0; width: 100%; height: 312px;" allowfullscreen></iframe>
            """,
            unsafe_allow_html=True,
        )
