FROM python:3.8

RUN pip install -U pandas mlflow boto3 sklearn
RUN mkdir -p /app

ARG DISABLE_CACHE
COPY test.py /app/

WORKDIR /app
ENTRYPOINT ["python", "test.py"]

# docker desktop open
# terminal cd 이동
# docker build -t sona0817/spaceship-titanic-test:v0.0.1 .
# docker build -t 도커아이디/경로:버전 띄어쓰기 . (현재 경로을 알려줌)
# docker push sona0817/spaceship-titanic-test:v0.0.1