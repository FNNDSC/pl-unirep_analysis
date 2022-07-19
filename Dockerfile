FROM tensorflow/tensorflow:1.3.0-gpu-py3
WORKDIR /usr/local/src/pl-unirep_analysis


COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . . 
ARG extras_require=none
RUN pip install .

COPY docker-entrypoint.sh chris_plugin_info.json /
CMD ["unirep_analysis"]
