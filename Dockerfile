# FROM continuumio/miniconda3:4.10.3-alpine
FROM python:3.7

# RUN apt-get -y update && apt-get -y install build-essential
# RUN conda update -n base -c defaults conda
# RUN conda install --force-reinstall -y -q --name base pip

# chown changes owner from root owner (1000) to the first user inside the env (100)
# COPY --chown=1000:100 requirements.txt /opt/requirements.txt
# COPY . .
# # RUN conda install --force-reinstall -y -q --name base -c conda-forge --file requirements.txt
# # RUN conda install --force-reinstall -y -q --name base pip
# RUN pip install -r requirements.txt
# CMD streamlit run ./app.py

COPY . /var/app/
WORKDIR /var/app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD streamlit run ./app.py
