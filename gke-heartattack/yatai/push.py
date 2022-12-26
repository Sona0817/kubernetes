import argparse
import subprocess
import shlex

from load_model import load
import bentoml

from sklearn import svm
from sklearn import datasets

def bento_serve(opt):
    # model = load(model_name=opt.model_name, version=opt.model_version)
    # bentoml.sklearn.save_model("adaboost", model)

    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # Model Training
    clf = svm.SVC(gamma='scale')
    clf.fit(X, y)
    bentoml.sklearn.save_model("iris_clf", clf)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, help='MLflow model name')
    parser.add_argument('--model-version', type=int, help='MLFlow model version')
    parser.add_argument('--api-token', type=str, help='MLFlow model version')
    opt = parser.parse_args()

    bento_serve(opt)
    subprocess.run(["chmod", "+x", "bentoml_command.sh"])
    subprocess.call(shlex.split(f"./bentoml_command.sh {opt.api_token} http://34.71.200.188"))

    while 1:
        True
