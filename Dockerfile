###############################################################################
# main
###############################################################################

FROM continuumio/miniconda3:4.8.2 AS main

# RUN apt-get -y update && \
#     apt-get -y install build-essential
RUN conda update -n base -c defaults conda

# chown changes owner from root owner (1000) to the first user inside the env (100)
# COPY --chown=1000:100 requirements.txt /opt/requirements.txt
# RUN conda install --force-reinstall -y -q --name base -c conda-forge --file /opt/requirements.txt
RUN conda install --force-reinstall -y -q --name base pip

COPY . /var/app/
# WORKDIR /var/dev
WORKDIR /var/app
RUN pip install -r dev-requirements.txt
CMD streamlit run ./app.py

###############################################################################
# test
###############################################################################

FROM main AS test
COPY . /var/dev/
WORKDIR /var/dev
# add unit test instruction here: RUN xxxxxx
# add integration test instruction here: RUN xxxxx
