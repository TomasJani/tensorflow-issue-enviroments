# https://github.com/tensorflow/tensorflow/issues/47263
FROM python:3.8.5-slim as base

WORKDIR /test

COPY script.py .

RUN python3 -m pip install --upgrade pip  && \
    python3 -m pip install tf-nightly-gpu --user

CMD [ "script.py" ]
ENTRYPOINT [ "python3" ]
