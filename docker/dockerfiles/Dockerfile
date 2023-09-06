ARG TENSORFLOW_BASE=tensorflow/tensorflow:2.11.0-gpu
FROM ${TENSORFLOW_BASE} as base_img

ARG TENSORFLOW_BASE
RUN echo "" && echo "Friendly info: this docker uses the base image: ${TENSORFLOW_BASE}" && echo ""

# python install from pip3
COPY docker/requirements/requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt && \
    rm /requirements.txt
RUN pip3 uninstall nvidia_cublas_cu11 -y

# Make data and libs accressible
COPY data /model/data
COPY run /model/run
COPY utils /model/utils
RUN mkdir -p /model/result
ENV PYTHONPATH=$PYTHONPATH:/model

# Define workdir
WORKDIR /model

# Define the correct entrypoint
ENTRYPOINT []
