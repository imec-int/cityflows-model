FROM continuumio/miniconda3:4.9.2

WORKDIR /app

COPY environment-docker.yml .
RUN conda env create -f environment-docker.yml

COPY src/config* src/
COPY src/model src/model
COPY src/service_utils src/service_utils
COPY src/service.py src/service.py
COPY src/utils src/utils
COPY run.sh .

ENTRYPOINT [ "./run.sh" ]
