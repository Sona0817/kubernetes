import argparse
import os
import pandas as pd
import mlflow

def test(root_path, model):
    test = pd.read_csv(os.path.join(root_path, 'dataset', 'test.csv'))
    test_preds = model.predict(test)

    print(test_preds)

def main(opt):
    mlflow.set_tracking_uri("http://mlflow-server-service.mlflow-system.svc:5000")
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio-service.kubeflow.svc:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"

    # model = mlflow.lightgbm.load_model(opt.model_path)
    model = mlflow.pyfunc.load_model(opt.model_path)
    test(opt.data_path, model)
    print('Test Completed')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, help='dataset root path')
    parser.add_argument('--model-path', type=str, help='model path in mlflow, i,e. s3://~')

    opt = parser.parse_args()
    main(opt)

