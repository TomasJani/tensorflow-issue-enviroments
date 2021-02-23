# https://github.com/tensorflow/tensorflow/issues/47264
FROM python:3.7.9-slim as base

WORKDIR /test

COPY script.py .

RUN python3 -m pip install --upgrade pip  && \
    python3 -m pip install tensorflow==2.4.1 --user

CMD [ "script.py" ]
ENTRYPOINT [ "python3" ]
