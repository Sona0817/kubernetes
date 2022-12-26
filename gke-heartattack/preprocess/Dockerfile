FROM python:3.8

RUN pip install -U sklearn lazypredict matplotlib seaborn pandas numpy gcsfs
RUN mkdir -p /app
ENV PYTHONUNBUFFERED 1

COPY ./preprocess/preprocess.py /app/

WORKDIR /app


