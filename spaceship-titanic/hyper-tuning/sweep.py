import argparse
import os
import time
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

import wandb

def configure():
    sweep_config = \
    {'method': 'random',
     'metric': {'goal': 'maximize', 'name': 'MEAN ACCURACY'},
     'parameters': {'n_estimators': {'values': [10, 50, 100, 200]},
                    'max_depth': {'values': [6, 8, 10, 12, 14]},
                    'min_samples_leaf': {'values': [2, 8, 12, 18]},
                    'min_samples_split': {'values': [2, 8, 16, 20]}
                    }
     }

    return sweep_config


def main(hyperparameters=None):
    wandb.init(project='spaceship-titanic', config=hyperparameters)
    config = wandb.config

    train = pd.read_csv(os.path.join(opt.data_path, 'dataset', 'train.csv'))
    TARGET = 'Transported'
    RANDOM_STATE = 12
    FOLDS = 5

    scores = []
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
        print(f"Fold={fold + 1}, Accuracy score: {acc:.2f}%, Run Time: {run_time:.2f}s")

    wandb.log({"MEAN ACCURACY": np.mean(scores)})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, help='dataset root path')
    opt = parser.parse_args()


    wandb.login(key='f493c713aff0996c47dcb4fb4eda80c4ca96a6d8')
    hyperparameters = configure()
    sweep_id = wandb.sweep(hyperparameters, project='spaceship-titanic')

    wandb.agent(sweep_id, main, count=10)