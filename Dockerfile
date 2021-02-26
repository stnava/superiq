FROM python:3.7-slim-buster
LABEL maintainer="tgosselin"

RUN apt-get update && \
    apt-get install -y build-essential cmake pkg-config git

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

RUN pip install numpy keras boto3 
RUN pip install --upgrade tensorflow tensorflow-probability 

ARG antspy_hash
RUN pip install git+https://github.com/ANTsX/ANTsPy.git@$antspy_hash
#RUN python ext/get_commit.py antspy $antspy_hash
#RUN pip install ext/antspyx* 

ARG antspynet_hash
RUN pip install git+https://github.com/ANTsX/ANTsPyNet.git@$antspynet_hash
#RUN python ext/get_commit.py antspynet $antspynet_hash
#RUN pip install ext/antspynet* 

# Package needs to be public first
#ARG superiq_hash
#RUN python -m pip install git+https://github.com/stnava/superiq.git@$superiq_hash

COPY . src
WORKDIR src
RUN python setup.py install

#ENV TF_NUM_INTEROP_THREADS=8
#ENV TF_NUM_INTRAOP_THREADS=8
#ENV ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8
