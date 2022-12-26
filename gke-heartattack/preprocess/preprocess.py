import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from lazypredict.Supervised import LazyClassifier

import base64
import json
from io import BytesIO


def preprocessing():
    heart = pd.read_csv(opt.gcs_path)
    heart.drop_duplicates(keep='first', inplace=True)
    # new shape
    print('Number of rows are', heart.shape[0], 'and number of columns are ', heart.shape[1])

    # shuffle dataset & split save
    df = heart.sample(frac=1, random_state=42).reset_index(drop=True)
    df[:int(df.shape[0] * 0.8)].to_csv(f'{opt.pvc_root_path}/train.csv', index=False) # save at kube pvc
    df[int(df.shape[0] * 0.8):].to_csv(f'{opt.pvc_root_path}/test.csv', index=False) # save at kube pvc

    return heart

def search_model(df):

    X = df.iloc[:, 1:-1]
    y = df["output"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=65)

    MM_scaler = MinMaxScaler()
    X_train = MM_scaler.fit_transform(X_train)
    X_test = MM_scaler.fit_transform(X_test)

    clf = LazyClassifier(verbose=0,
                         ignore_warnings=True,
                         custom_metric=None,
                         predictions=False,
                         random_state=12,
                         classifiers='all')

    _, predictions = clf.fit(X_train, X_test, y_train, y_test)

    return predictions


def visualize(pred):
    k = 7
    temp = pred[:k].round(5) * 100

    # Plot accuracy for different models
    plt.figure(figsize=(10, 4))
    ACC = sns.barplot(y=temp.index, x=temp["Accuracy"], label="Accuracy", edgecolor="violet", linewidth=3, orient="h",
                      palette="twilight_r")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Comparison")
    plt.xlim(80, 98)

    ACC.spines['left'].set_linewidth(3)
    for w in ['right', 'top', 'bottom']:
        ACC.spines[w].set_visible(False)

    # Write text on barplots
    k = 0
    for ACC in ACC.patches:
        width = ACC.get_width()
        plt.text(width + 0.1, (ACC.get_y() + ACC.get_height() - 0.3), s="{}%".format(temp["Accuracy"][k]),
                 fontname='monospace', fontsize=14, color='violet')
        k += 1

    plt.legend(loc="lower right")
    plt.tight_layout()

    tmpfile = BytesIO()
    plt.savefig(tmpfile, format="png")
    encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")

    html = f"<img src='data:image/png;base64,{encoded}'>"
    metadata = {
        "outputs": [
            {
                "type": "web-app",
                "storage": "inline",
                "source": html,
            },
        ],
    }

    with open("/mlpipeline-ui-metadata.json", "w") as html_writer:
        json.dump(metadata, html_writer)







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gcs-path', type=str, help='GCS path, gs://~') # data path
    parser.add_argument('--pvc-root-path', type=str, default='/home/sona', help='PVC root path') # pvc path for save
    opt = parser.parse_args()

    df = preprocessing()
    pred = search_model(df)
    visualize(pred)
