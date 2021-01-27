FROM python:3.7-slim-buster
LABEL maintainer="tgosselin"

RUN apt-get update && \
    apt-get install -y build-essential cmake pkg-config git

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

RUN pip install numpy keras boto3 
RUN pip install --upgrade tensorflow tensorflow-probability 

ADD ext ext
RUN python ext/get_latest.py 
RUN pip install ext/antspyx* 
RUN pip install ext/antspynet* 

COPY . src
WORKDIR src
RUN python setup.py install

ENV TF_NUM_INTEROP_THREADS=8
ENV TF_NUM_INTRAOP_THREADS=8
ENV ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8