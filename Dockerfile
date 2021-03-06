
# Docker file for unirep_analysis ChRIS plugin app
#
# Build with
#
#   docker build -t <name> .
#
# For example if building a local version, you could do:
#
#   docker build -t local/pl-unirep_analysis .
#
# In the case of a proxy (located at 192.168.13.14:3128), do:
#
#    docker build --build-arg http_proxy=http://192.168.13.14:3128 --build-arg UID=$UID -t local/pl-unirep_analysis .
#
# To run an interactive shell inside this container, do:
#
#   docker run -ti --entrypoint /bin/bash local/pl-unirep_analysis
#
# To pass an env var HOST_IP to container, do:
#
#   docker run -ti -e HOST_IP=$(ip route | grep -v docker | awk '{if(NF==11) print $9}') --entrypoint /bin/bash local/pl-unirep_analysis
#
FROM tensorflow/tensorflow:1.3.0-py3 as unirep
LABEL maintainer="FNNDSC <dev@babyMRI.org>"

WORKDIR /usr/local/src


COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . . 
RUN pip install .

COPY docker-entrypoint.sh /
CMD ["unirep_analysis"]
