# https://github.com/tensorflow/tensorflow/issues/47340
FROM python:3.8.5-slim as base

WORKDIR /test

COPY script.py .

RUN python3 -m pip install --upgrade pip  && \
    python3 -m pip install tensorflow==2.4.1 --user

CMD [ "script.py" ]
ENTRYPOINT [ "python3" ]
