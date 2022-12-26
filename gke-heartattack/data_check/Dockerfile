FROM python:3.8

RUN pip install -U pandas gcsfs
RUN mkdir -p /app
ENV PYTHONUNBUFFERED 1

COPY ./data_check/data_check.py /app/

WORKDIR /app


