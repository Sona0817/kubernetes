import argparse
import os
import time
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier

from lazypredict.Supervised import LazyClassifier

from mlflow.sklearn import save_model
from mlflow.tracking.client import MlflowClient
from mlflow.models.signature import infer_signature


def modeling(X_train, X_test, y_train, y_test):
    clf = LazyClassifier(verbose=0,
                         ignore_warnings=True,
                         custom_metric=None,
                         predictions=False,
                         random_state=12,
                         classifiers='all')

    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    print(f"Best5 Model: {[f'{idx + 1}.' + i for idx, i in enumerate(models[:5].index.values)]}")

def RF_Classifier():
    train = pd.read_csv(os.path.join(opt.data_path, 'dataset', 'train.csv'))
    test = pd.read_csv(os.path.join(opt.data_path, 'dataset', 'test.csv'))

    config = {'n_estimators': opt.n_estimators,
              'max_depth': opt.max_depth,
              'min_samples_leaf': opt.min_samples_leaf,
              'min_samples_split': opt.min_samples_split
              }

    TARGET = 'Transported'
    RANDOM_STATE = 12
    FOLDS = 5

    predictions = 0
    scores = []
    fimp = []
    RF_FEATURES = list(train.columns)[:-1]
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (train_idx, valid_idx) in enumerate(skf.split(train[RF_FEATURES], train[TARGET])):
        print(f'\033[94m')
        print(10 * "=", f"Fold={fold + 1}", 10 * "=")
        start_time = time.time()

        X_train, X_valid = train.iloc[train_idx][RF_FEATURES], train.iloc[valid_idx][RF_FEATURES]
        y_train, y_valid = train[TARGET].iloc[train_idx], train[TARGET].iloc[valid_idx]

        model = RandomForestClassifier(**config)
        model.fit(X_train, y_train)

        preds_valid = model.predict(X_valid)
        acc = accuracy_score(y_valid, preds_valid)
        scores.append(acc)
        run_time = time.time() - start_time

        fim = pd.DataFrame(index=RF_FEATURES,
                           data=model.feature_importances_,
                           columns=[f'{fold}_importance'])
        fimp.append(fim)

        print(f"Fold={fold + 1}, Accuracy score: {acc:.2f}%, Run Time: {run_time:.2f}s")
        test_preds = model.predict(test[RF_FEATURES])
        predictions += test_preds / FOLDS
    print("")
    print("Mean Accuracy :", np.mean(scores))

    return model, train[RF_FEATURES]


def upload_model_to_mlflow(model_name, model, train):

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio-service.kubeflow.svc:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
    client = MlflowClient("http://mlflow-server-service.mlflow-system.svc:5000")

    signature = infer_signature(train, model.predict(train))

    original = pd.read_csv(os.path.join(opt.data_path, 'train.csv'))
    input_example = original.sample(1)
    save_model(
        sk_model=model,
        path=model_name,
        serialization_format="cloudpickle",
        signature=signature,
        input_example=input_example
    )

    tags = {"Machin Learning": "spaceship titanic classification"}
    run = client.create_run(experiment_id="5", tags=tags) # mlflow dashboard check http://116.47.188.227:30842/
    client.log_artifact(run.info.run_id, model_name)

def load_data(root_path):
    X_train = pd.read_csv(os.path.join(root_path, 'dataset', 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(root_path, 'dataset', 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(root_path, 'dataset', 'y_train.csv'))
    y_test = pd.read_csv(os.path.join(root_path, 'dataset', 'y_test.csv'))

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, help='dataset root path')
    parser.add_argument('--model-name', type=str, help='name of model')
    parser.add_argument('--n-estimators', type=int, help='The number of trees in the forest.')
    parser.add_argument('--max-depth', type=int, help='The maximum depth of the tree.')
    parser.add_argument('--min-samples-leaf', type=int, help='The minimum number of samples required to be at a leaf node.')
    parser.add_argument('--min-samples-split', type=int, help='The minimum number of samples required to split an internal node.')

    opt = parser.parse_args()

    X_train, X_test, y_train, y_test = load_data(opt.data_path)
    modeling(X_train, X_test, y_train, y_test)
    model, train = RF_Classifier()

    upload_model_to_mlflow(opt.model_name, model, train)


