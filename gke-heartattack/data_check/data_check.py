import argparse
import pandas as pd

def data_checking():
    heart = pd.read_csv(opt.gcs_path)
    print('Number of rows are', heart.shape[0], 'and number of columns are ', heart.shape[1])

    # check null values
    null_list = [(f'{value} null values in {idx} column') for idx, value in heart.isnull().sum().iteritems() if value != 0]
    for i in null_list: print(i)

    # check duplicated values
    duplicated_list = list(heart[heart.duplicated()].index)
    print(f"duplicated {len(duplicated_list)} rows! \nduplicated index: {duplicated_list}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gcs-path', type=str, help='GCS path, gs://~')
    opt = parser.parse_args()

    data_checking()
