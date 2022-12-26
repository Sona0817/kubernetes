import argparse
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns

from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_recall_fscore_support as score

import mlflow
from mlflow.tracking import MlflowClient

import base64
import json
from io import BytesIO


def test(model):
    test_df = pd.read_csv(opt.pvc_root_path+'/test.csv')
    FEATURES = list(test_df.columns)[1:-1]
    TARGET = "output"

    y_test = test_df[TARGET]

    start = time.time()
    y_pred = model.predict(test_df[FEATURES])
    end = time.time()

    print(f"Execution time of model: {round((end-start), 5)} seconds")
    # Plot and compute metric
    compute(y_pred, y_test)


def load_model_from_mlflow():
    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio-service.kubeflow.svc:9000"
    client = MlflowClient("http://mlflow-service.mlflow-system.svc:5000")

    filter_string = f"name='{opt.model_name}'"
    results = client.search_model_versions(filter_string)  # 버전별로 따로 나옴

    for res in results:
        if res.version == str(opt.model_version):
            model_uri = res.source
            break

    model = mlflow.pyfunc.load_model(model_uri)

    return model

def compute(Y_pred,Y_test):
    #Output plot
    plt.figure(figsize=(8,3))
    plt.scatter(range(len(Y_pred)),Y_pred,color="cornflowerblue",lw=5,label="Predictions")
    plt.scatter(range(len(Y_test)),Y_test,color="darkorange",label="Actual")
    plt.title("Prediction Values vs Real Values")
    plt.legend()

    tmpfile = BytesIO()
    plt.savefig(tmpfile, format="png")
    encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")
    html1 = f"<img src='data:image/png;base64,{encoded}'>" # matpoltlib to html

    plt.figure(figsize=(8, 3))
    cm=confusion_matrix(Y_test,Y_pred)
    class_label = ["High-risk", "Low-risk"]
    df_cm = pd.DataFrame(cm, index=class_label,columns=class_label)
    sns.heatmap(df_cm,annot=True,cmap='Pastel1',linewidths=2,fmt='d')
    plt.title("Confusion Matrix",fontsize=15)
    plt.xlabel("Predicted")
    plt.ylabel("True")


    tmpfile = BytesIO()
    plt.savefig(tmpfile, format="png")
    encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")

    html2 = f"<img src='data:image/png;base64,{encoded}'>"
    metadata = {
        "outputs": [
            {
                "type": "web-app",
                "storage": "inline",
                "source": [html1, html2],
            },
        ],
    }

# must fixed name "/mlpipeline-ui-metadata.json"
    with open("/mlpipeline-ui-metadata.json", "w") as html_writer:
        json.dump(metadata, html_writer)

    #Calculate Metrics
    acc=accuracy_score(Y_test,Y_pred)
    mse=mean_squared_error(Y_test,Y_pred)
    precision, recall, fscore, train_support = score(Y_test, Y_pred, pos_label=1, average='binary')
    print('Precision: {} \nRecall: {} \nF1-Score: {} \nAccuracy: {} %\nMean Square Error: {}'.format(
        round(precision, 3), round(recall, 3), round(fscore,3), round((acc*100),3), round((mse),3)))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pvc-root-path', type=str, default='/home/jeff', help='PVC root path')
    parser.add_argument('--model-name', type=str, help='name of model')
    parser.add_argument('--model-version', type=int, help='version of model')

    opt = parser.parse_args()


    model = load_model_from_mlflow()
    test(model)
