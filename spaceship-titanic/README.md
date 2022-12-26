# Spaceship titanic model in kubernetes
작성자 : sona0817

우주선 타이타닉호에서 살아남은 사람 예측 모델 전처리, 파라미터튜닝, 트레인, 테스트, 인퍼런스, 쿠버네티스를 통한 모델 배포 과정을 담은 python파일 및 yaml파일

## 1. 전처리
- preprocessing.py 파일 작성
- 같은 폴더에 Dockerfile 작성  
  도커허브에 올릴 이미지를 작성하는 파일로, 파일명은 무조건 Dockerfile이여야 한다.  
  따라서 도커파일이 필요한 파이썬 파일별로 폴더를 나누어주는 것이 좋다.
- 도커 빌드 및 푸시
- pipeline.py 작성
- pipeline.py 실행하여 yaml파일 생성
- kubeflow dashbord에 pipeline.yaml 업로드
- create run

## 2. 파라미터 튜닝
- hyper_parameter_tuning.py 작성
- 같은 폴더에 Dockerfile 작성
- 도커 빌드 및 푸시
- pipeling.py 작성
- yaml파일 생성
- kubeflow 에 yaml 업로드
- create run

## 3. 트레인
- 위와 동일한 순서로 진행, 여기서 모델이 생성됨!  
  def upload_model_to_mlflow()를 확인
- 생성된 모델은 mlflow dashoboard에서 확인 가능
- mlflow에 등록된 주소를 test할 때 model_path에 작성

## 4. 테스트
- 위와 동일한 순서로 진행
- kubeflow 실행 시 파라미터에 model_path를 mlflow에 등록된 주소로 작성

## 5. 인퍼런스
- 도커 빌드 및 푸시까지만 진행
- server.yaml 파일 직접 작성
- 로컬에서 .py로 실행할 수 있도록 하는 것이 아니라 kubeflow를 통해 배포할 예정이기 때문에 1~4과정을 하나도 넣어줄 과정 필요
- 이 때 bento 라이브러리 사용
- ClusterIP의 포트번호 확인은  
  kubens istio-system  
  k get svc 를 통해 확인 가능
- k port-forward svc/서비스이름 아무거나포트번호:컨테이너포트번호
  k port-forward svc/spaceship 8775:5000
- k apply -f server.yaml 실행
- inferenc.py #CLI post command 코드를 터미널에서 실행
