FROM python:3.8

RUN pip install -U pandas mlflow boto3 sklearn bentoml protobuf~=3.19.0
RUN mkdir -p /app

ARG DISABLE_CACHE
COPY bento_packer.py /app/
COPY bento_service.py /app/
COPY load_model.py /app/

WORKDIR /app
ENTRYPOINT ["python", "bento_packer.py"]

# docker desktop open
# terminal cd 이동
# docker build -t sona0817/spaceship-titanic-test:v0.0.1 .
# docker build -t 도커아이디/경로:버전 띄어쓰기 . (현재 경로을 알려줌)
# docker push sona0817/spaceship-titanic-test:v0.0.1