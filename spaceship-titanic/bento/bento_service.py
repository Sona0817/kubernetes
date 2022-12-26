import pandas as pd
from bentoml import BentoService, env, api, artifacts
from bentoml.adapters import DataframeInput
from bentoml.frameworks.sklearn import SklearnModelArtifact

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


@artifacts([SklearnModelArtifact('rf')])
@env(pip_packages=["scikit-learn"])
class Spaceship_Service(BentoService):
    @api(input=DataframeInput(), batch=True)
    def predict(self, df: pd.DataFrame):
        # Preprocess
        df.columns = ["PassengerId", "HomePlanet", "CryoSleep", "Cabin","Destination",
                      "Age","VIP","RoomService","FoodCourt","ShoppingMall","Spa",
                      "VRDeck","Name"]

        print('THIS IS INPUT DF: ', df)

        df.drop(["PassengerId"], axis=1, inplace=True)
        imputer_cols = ["Age", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "RoomService"]
        imputer = SimpleImputer(strategy='median')
        imputer.fit(df[imputer_cols])

        df[imputer_cols] = imputer.transform(df[imputer_cols])
        df["HomePlanet"].fillna('Z', inplace=True)

        label_cols = ["HomePlanet", "CryoSleep", "Cabin", "Destination", "VIP"]

        for col in label_cols:
            df[col] = df[col].astype(str)
            df[col] = LabelEncoder().fit_transform(df[col])

        # df.drop(["Name", "Cabin"], axis=1, inplace=True)
        FEATURES = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService',
        'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

        return self.artifacts.rf.predict(df[FEATURES])
