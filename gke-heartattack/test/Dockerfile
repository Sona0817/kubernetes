FROM python:3.8

RUN pip install -U sklearn pandas seaborn mlflow boto3
RUN mkdir -p /app
ENV PYTHONUNBUFFERED 1

COPY ./test/test.py /app/

WORKDIR /app


