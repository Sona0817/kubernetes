import argparse
import os
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


def preprocess_data(data_path):
    train = pd.read_csv(os.path.join(data_path, 'train.csv'))
    test = pd.read_csv(os.path.join(data_path, 'test.csv'))

    STRATEGY = 'median'

    train.drop(["PassengerId"], axis=1, inplace=True)
    test.drop(["PassengerId"], axis=1, inplace=True)
    TARGET = 'Transported'

    # Imputing Missing Values
    imputer_cols = ["Age", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "RoomService"]
    imputer = SimpleImputer(strategy=STRATEGY)
    imputer.fit(train[imputer_cols])
    train[imputer_cols] = imputer.transform(train[imputer_cols])
    test[imputer_cols] = imputer.transform(test[imputer_cols])
    train["HomePlanet"].fillna('Z', inplace=True)
    test["HomePlanet"].fillna('Z', inplace=True)


    label_cols = ["HomePlanet", "CryoSleep","Cabin", "Destination" ,"VIP"]
    train, test = label_encoder(train, test, label_cols)

    split_dataset(train, test, TARGET, data_path)


def label_encoder(train,test,columns):
    for col in columns:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
        train[col] = LabelEncoder().fit_transform(train[col])
        test[col] =  LabelEncoder().fit_transform(test[col])
    return train, test


def split_dataset(train, test, TARGET, data_path):
    train.drop(["Name", "Cabin"], axis=1, inplace=True)
    test.drop(["Name", "Cabin"], axis=1, inplace=True)
    X = train.drop(TARGET, axis=1)
    y = train[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        random_state=12,
                                                        test_size=0.33)

    os.makedirs(os.path.join(data_path, 'dataset'), exist_ok=True)
    train.to_csv(os.path.join(data_path, 'dataset', 'train.csv'), index=False)
    test.to_csv(os.path.join(data_path, 'dataset', 'test.csv'), index=False)

    X_train.to_csv(os.path.join(data_path, 'dataset', 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(data_path, 'dataset', 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(data_path, 'dataset', 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(data_path, 'dataset', 'y_test.csv'), index=False)
    print('Preprocess Completed')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, help='dataset root path')
    opt = parser.parse_args()

    preprocess_data(opt.data_path)