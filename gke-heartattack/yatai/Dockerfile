FROM python:3.8

RUN pip install -U numpy sklearn mlflow boto3 bentoml==1.0.0
RUN mkdir -p /app
ENV PYTHONUNBUFFERED 1

COPY ./yatai /app/

WORKDIR /app


