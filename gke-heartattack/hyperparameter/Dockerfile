FROM python:3.8

RUN pip install -U numpy pandas scikit-learn wandb
RUN mkdir -p /app
ENV PYTHONUNBUFFERED 1

COPY ./hyperparameter/hyperparameter.py /app/
WORKDIR /app
