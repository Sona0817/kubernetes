import kfp
from kfp import dsl
from kfp import onprem

def data_check_op(data_path):

    return dsl.ContainerOp(
        name='Data Checking', # container name
        image='gcr.io/galvanized-yeti-356902/heart-data_check:latest', # docker image latest
        command=["python", "data_check.py", '--gcs-path', data_path],
    )

def preprocess_op(pvc_name, volume_name, volume_mount_path, data_path):

    return dsl.ContainerOp(
        name='Preprocessing',
        image='gcr.io/galvanized-yeti-356902/heart-preprocess:latest',
        command=["python", "preprocess.py", '--gcs-path', data_path, '--pvc-root-path', volume_mount_path],
        output_artifact_paths={'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'},
    ).apply(onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path))

def hyperparameter_op(pvc_name, volume_name, volume_mount_path, sweep_count):

    return dsl.ContainerOp(
        name='Hyperparameter Tuning',
        image='gcr.io/galvanized-yeti-356902/heart-hyperparameter:latest',
        command=["python", "hyperparameter.py", '--pvc-root-path', volume_mount_path, '--sweep-count', sweep_count],
        output_artifact_paths={'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'},
    ).apply(onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path))

def train_op(pvc_name, volume_name, volume_mount_path, repo_name):

    return dsl.ContainerOp(
        name='Train Model',
        image='gcr.io/galvanized-yeti-356902/heart-train:latest',
        command=["python", "train.py", '--pvc-root-path', volume_mount_path, '--repo-name', repo_name],
    ).apply(onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path))

def test_op(pvc_name, volume_name, volume_mount_path, model_name, model_version):

    return dsl.ContainerOp(
        name='Test Model',
        image='gcr.io/galvanized-yeti-356902/heart-test:latest',
        command=["python", "test.py", '--pvc-root-path', volume_mount_path, '--model-name', model_name, '--model-version', model_version],
        output_artifact_paths={'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'},
    ).apply(onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path))

def yatai_op(model_name, model_version, api_token):

    return dsl.ContainerOp(
        name='Push to Yatai',
        image='gcr.io/galvanized-yeti-356902/heart-yatai:latest',
        command=["python", "push.py", '--model-name', model_name, '--model-version', model_version, '--api-token', api_token],
    )

@dsl.pipeline(
    name='Heart Attack Pipeline',
    description=''
)
def gke_pipeline(Preprocessing: bool,
                 Hyperparamer_Tuning: bool,
                 Train: bool,
                 Test: bool,
                 Yatai: bool,
                 HYPER_sweep_count: int,
                 TRAIN_repo_name: str,
                 TEST_model_name: str,
                 TEST_model_version: int,
                 YATAI_model_name: str,
                 YATAI_model_version: int,
                 YATAI_api_token: str,
                 ):
    pvc_name = "heart-attack"
    volume_name = 'pipeline'
    volume_mount_path = '/home/sona' # data container path
    GCS_data_path = 'gs://heart-attack-dataset-sona/heart.csv' # data path

    with dsl.Condition(Preprocessing == True):
        _data_check_op = data_check_op(GCS_data_path)
        _preprocess_op = preprocess_op(pvc_name, volume_name, volume_mount_path, GCS_data_path).after(_data_check_op)

    with dsl.Condition(Hyperparamer_Tuning == True):
        _hyperparameter_op = hyperparameter_op(pvc_name, volume_name, volume_mount_path, HYPER_sweep_count).after(_preprocess_op)

    with dsl.Condition(Train == True):
        _train_op = train_op(pvc_name, volume_name, volume_mount_path, TRAIN_repo_name)

    with dsl.Condition(Test == True):
        _test_op = test_op(pvc_name, volume_name, volume_mount_path, TEST_model_name, TEST_model_version).after(_train_op)

    with dsl.Condition(Yatai == True):
        _yatai_op = yatai_op(YATAI_model_name, YATAI_model_version, YATAI_api_token).after(_test_op)


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(gke_pipeline, './pipeline.yaml')
