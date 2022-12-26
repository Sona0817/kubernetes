FROM python:3.8

RUN pip install -U pandas scikit-learn
RUN mkdir -p /app

ARG DISABLE_CACHE
COPY preprocessing.py /app/

WORKDIR /app
ENTRYPOINT ["python", "preprocessing.py"]

# docker desktop open
# terminal cd 이동
# docker build -t sona0817/spaceship-titanic-hyp:v0.0.1 .
# docker push sona0817/spaceship-titanic-hyp:v0.0.1