# https://github.com/tensorflow/tensorflow/issues/47289
FROM python:3.7.6-slim as base

WORKDIR /test

COPY script.py .

RUN python3 -m pip install --upgrade pip  && \
    python3 -m pip install tensorflow==2.1.0 --user

RUN [ "script.py" ]
ENTRYPOINT [ "python3" ]
