# https://github.com/tensorflow/tensorflow/issues/47242
# Does not work, tf-2.4.1 does not support py-3.9.1
FROM python:3.9.1-slim as base

WORKDIR /test

COPY script.py .

RUN python3 -m pip install --upgrade pip  && \
    python3 -m pip install tensorflow==2.4.1 --user

CMD [ "script.py" ]
ENTRYPOINT [ "python3" ]
