import argparse
import os
import pandas as pd
import numpy as np
import time
import json

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from mlflow.tracking.client import MlflowClient
from mlflow.models.signature import infer_signature
from mlflow.sklearn import save_model


def training():

    df = pd.read_csv(opt.pvc_root_path+'/train.csv')

    with open(f"{opt.pvc_root_path}/best_hyperparameter.json", "r") as json_file:
        config = json.load(json_file) # as dic

    FEATURES = list(df.columns)[1:-1]
    TARGET = "output"
    RANDOM_STATE = 42
    FOLDS = 5

    scores = []
    fimp = []
    MM_scaler = MinMaxScaler()
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (train_idx, valid_idx) in enumerate(skf.split(df[FEATURES], df[TARGET])):
        print(f'\033[94m')
        print(10 * "=", f"Fold={fold + 1}", 10 * "=")
        start_time = time.time()

        X_train, X_valid = df.iloc[train_idx][FEATURES], df.iloc[valid_idx][FEATURES]
        y_train, y_valid = df[TARGET].iloc[train_idx], df[TARGET].iloc[valid_idx]

        X_train = MM_scaler.fit_transform(X_train)
        X_valid = MM_scaler.fit_transform(X_valid)

        model =AdaBoostClassifier(**config) ## ** > release dic
        model.fit(X_train, y_train)

        preds_valid = model.predict(X_valid)
        acc = accuracy_score(y_valid, preds_valid)
        scores.append(acc)
        run_time = time.time() - start_time

        fim = pd.DataFrame(index=FEATURES,
                           data=model.feature_importances_,
                           columns=[f'{fold}_importance'])
        fimp.append(fim)

        print(f"Fold={fold + 1}, Accuracy score: {acc:.2f}%, Run Time: {run_time:.2f}s")

    print("")
    print("Mean Accuracy :", np.mean(scores))

    return model, df[FEATURES]


def upload_model_to_mlflow(model, data):

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio-service.kubeflow.svc:9000" # service name.namespace.svc:portnumber
    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
    client = MlflowClient("http://mlflow-service.mlflow-system.svc:5000")

    signature = infer_signature(data, model.predict(data))

    original = pd.read_csv(os.path.join(opt.pvc_root_path, 'train.csv'))
    input_example = original.sample(1)
    save_model(
        sk_model=model,
        path=opt.repo_name,
        serialization_format="cloudpickle",
        signature=signature,
        input_example=input_example
    )

    experiment_name = "heart attack experiments"
    current_experiment = client.get_experiment_by_name(experiment_name)

    # experiment가 있다면 사용하고 없으면 생성
    if current_experiment:
        experiment_id = dict(current_experiment)['experiment_id']
    else:
        experiment_id = client.create_experiment(experiment_name)

    tags = {"ML": "heart attack classification"}
    run = client.create_run(experiment_id=experiment_id, tags=tags)
    client.log_artifact(run.info.run_id, opt.repo_name)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pvc-root-path', type=str, default='/home/jeff', help='PVC root path')
    parser.add_argument('--repo-name', type=str, help='model repo name for artifact path') # mlflow folder name
    opt = parser.parse_args()

    model, data = training()
    upload_model_to_mlflow(model, data)