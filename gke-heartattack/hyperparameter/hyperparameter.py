import argparse
import pandas as pd
import numpy as np
import time

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

import wandb
import json

def configure():
    sweep_config = {'method': 'random',
                    'metric': {'goal': 'maximize', 'name': 'MEAN ACCURACY'},
                    'parameters': {'n_estimators': {'values': [10, 50, 100, 200]},
                                   'learning_rate': {'min': 0.001, 'max': 1, 'distribution': 'uniform'},
                                   'algorithm': {'values': ['SAMME', 'SAMME.R']},
                                   'random_state': {'value': 42}
                                   }
                    }
    return sweep_config


def parameter_tuning(hyperparameters=None):
    run = wandb.init(project='heart-attack', config=hyperparameters)
    config = wandb.config

    df = pd.read_csv(opt.pvc_root_path+'/train.csv')
    FEATURES = list(df.columns)[1:-1]
    TARGET = "output"
    RANDOM_STATE = 42
    FOLDS = 5
    scores = []

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

        model =AdaBoostClassifier(**config)
        model.fit(X_train, y_train)

        preds_valid = model.predict(X_valid)
        acc = accuracy_score(y_valid, preds_valid)
        scores.append(acc)
        run_time = time.time() - start_time
        print(f"Fold={fold + 1}, Accuracy score: {acc:.2f}%, Run Time: {run_time:.2f}s")

    wandb.log({"MEAN ACCURACY": np.mean(scores)})
    wandb.finish()

# show at kube dashboard as markdown
    metadata = {
        "outputs": [
            {
                "type": "markdown",
                "storage": "inline",
                "source": f"<iframe src=\"{run.get_sweep_url()}\" width=\"100%\" height=\"700\"/>"
            },
        ],
    }

# must fixed name "mlpipeline-ui-metadata.json"
    with open("/mlpipeline-ui-metadata.json", "w") as metadata_file:
        json.dump(metadata, metadata_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pvc-root-path', type=str, default='/home/jeff', help='PVC root path')
    parser.add_argument('--sweep-count', type=int, help='count of hyperparameter tuning')

    opt = parser.parse_args()

    wandb.login(key='f493c713aff0996c47dcb4fb4eda80c4ca96a6d8') # wandb>setting>dangerzone>copy
    hyperparameters = configure()


    sweep_id = wandb.sweep(hyperparameters, project='heart-attack')
    wandb.agent(sweep_id, parameter_tuning, count=opt.sweep_count)

    api = wandb.Api()
    sweep = api.sweep(f"sona0817/heart-attack/{sweep_id}") # change to wandb username
    best_run = sweep.best_run() # get best parameter as dictionary
    print("Best Metric")
    print(best_run.config)

    with open(f"{opt.pvc_root_path}/best_hyperparameter.json", "w") as json_file:
        json.dump(best_run.config, json_file)