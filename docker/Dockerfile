FROM jupyter/scipy-notebook

USER root
ENV GRANT_SUDO=yes

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    portaudio19-dev \
    python3-pyaudio && \
    rm -rf /var/lib/apt/lists/*

USER $NB_USER
ENV GRANT_SUDO=no

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR $HOME/work