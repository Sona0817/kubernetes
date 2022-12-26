FROM python:3.8

RUN pip install -U sklearn pandas numpy mlflow boto3
RUN mkdir -p /app
ENV PYTHONUNBUFFERED 1

COPY ./train/train.py /app/

WORKDIR /app


