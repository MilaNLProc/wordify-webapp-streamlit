FROM python:3.7

COPY . /var/app/
WORKDIR /var/app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD streamlit run ./app.py
