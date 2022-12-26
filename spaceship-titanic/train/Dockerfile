FROM python:3.8

RUN pip install -U numpy pandas scikit-learn lazypredict mlflow boto3
RUN mkdir -p /app

ARG DISABLE_CACHE
COPY train.py /app/

WORKDIR /app
ENTRYPOINT ["python", "train.py"]

# docker desktop open
# terminal cd 이동
# docker build -t sona0817/spaceship-titanic-train:v0.0.1 .
# docker build -t 도커아이디/경로:버전 띄어쓰기 . (현재 경로을 알려줌)
# docker push sona0817/spaceship-titanic-train:v0.0.1