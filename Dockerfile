from python:3.8-slim
ADD sources.list /etc/apt/sources.list
RUN apt -y update
RUN apt install -y build-essential cmake gfortran wget graphicsmagick libgraphicsmagick1-dev \
    libatlas-base-dev libavcodec-dev libavformat-dev libjpeg-dev liblapack-dev libswscale-dev pkg-config \
    libsm6 libx11-6 libxext6 libatlas3-base \
    && apt clean && rm -rf /tmp/* /var/tmp/*

ARG BRANCH=19.21
RUN wget -c -q http://dlib.net/files/dlib-${BRANCH}.tar.bz2 \
    && tar jxvf dlib-${BRANCH}.tar.bz2 \
    && cd dlib-${BRANCH} \
    && python setup.py install \
    && cd / \
    && rm -rf *.tar.bz2 /dlib-${BRANCH}

RUN pip install -U pip
RUN apt remove -y cmake gfortran wget graphicsmagick libgraphicsmagick1-dev libatlas-base-dev libavcodec-dev \
    libavformat-dev libjpeg-dev liblapack-dev libswscale-dev pkg-config \
    && apt autoremove -y && apt clean && rm -rf /tmp/* /var/tmp/*


ADD requirements.txt /requirements.txt
RUN pip install -r requirements.txt

COPY torch/resnet34-333f7ec4.pth /root/.cache/torch/hub/checkpoints/resnet34-333f7ec4.pth
ADD src /fairface
ADD dlib_models /fairface/dlib_models
ADD fair_face_models /fairface/fair_face_models
WORKDIR /fairface


CMD python app.py