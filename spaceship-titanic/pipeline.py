import kfp
from kfp import dsl
from kfp import onprem

def preprocess_op(pvc_name, volume_name, volume_mount_path):

    return dsl.ContainerOp(
        name='Preprocess Data',
        image='sona0817/spaceship-titanic-preprocessing:v0.0.1',
        arguments=['--data-path', volume_mount_path],
    ).apply(onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path))

def hyp_op(pvc_name, volume_name, volume_mount_path):

    return dsl.ContainerOp(
        name='Hyperparameter Tuning',
        image='sona0817/spaceship-titanic-hyp:v0.0.1',
        arguments=['--data-path', volume_mount_path],
    ).apply(onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path))

def train_op(pvc_name, volume_name, volume_mount_path,
             model_name, n_estimators, max_depth, min_samples_leaf, min_samples_split):

    return dsl.ContainerOp(
        name='Train Model',
        image='sona0817/spaceship-titanic-train:v0.0.1',
        arguments=['--data-path', volume_mount_path,
                   '--model-name', model_name,
                   '--n-estimators', n_estimators,
                   '--max-depth', max_depth,
                   '--min-samples-leaf', min_samples_leaf,
                   '--min-samples-split', min_samples_split,
                   ]
    ).apply(onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path))

def test_op(pvc_name, volume_name, volume_mount_path,
            model_path):

    return dsl.ContainerOp(
        name='Test',
        image='sona0817/spaceship-titanic-test:v0.0.2',
        arguments=['--data-path', volume_mount_path,
                   '--model-path', model_path,]
    ).apply(onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path))

def bento_op(pvc_name, volume_name, volume_mount_path,
             model_name, version):

    return dsl.ContainerOp(
        name='Bento Packing',
        image='sona0817/spaceship-titanic-bento:v0.0.1',
        arguments=['--data-path', volume_mount_path,
                   '--model-name', model_name,
                   '--version', version],
    ).apply(onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path))

@dsl.pipeline(
    name='Spaceship Titanic Pipeline',
    description=''
)
def _pipeline(PREPROCESS_yes_no: str,
             MODE_hyp_train_test_bento: str,
             TRAIN_model_name: str,
             TRAIN_n_estimators: int,
             TRAIN_max_depth: int,
             TRAIN_min_samples_leaf: int,
             TRAIN_min_samples_split: int,
             TEST_model_path: str,
             BENTO_model_name: str,
             BENTO_version: int):
    pvc_name = "workspace-demo"
    volume_name = 'pipeline'
    volume_mount_path = '/home/sona'
    with dsl.Condition(PREPROCESS_yes_no == 'yes'):
        _preprocess_op = preprocess_op(pvc_name, volume_name, volume_mount_path)

    with dsl.Condition(MODE_hyp_train_test_bento == 'hyp'):
        _hyp_op = hyp_op(pvc_name, volume_name, volume_mount_path).after(_preprocess_op)

    with dsl.Condition(MODE_hyp_train_test_bento == 'train'):
        _train_op = train_op(pvc_name, volume_name, volume_mount_path, TRAIN_model_name,
                             TRAIN_n_estimators, TRAIN_max_depth,
                             TRAIN_min_samples_leaf, TRAIN_min_samples_split).after(_preprocess_op)

    with dsl.Condition(MODE_hyp_train_test_bento == 'test'):
        _test_op = test_op(pvc_name, volume_name, volume_mount_path, TEST_model_path).after(_preprocess_op)

    with dsl.Condition(MODE_hyp_train_test_bento == 'bento'):
        _bento_op = bento_op(pvc_name, volume_name, volume_mount_path, BENTO_model_name, BENTO_version).after(_test_op)

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(_pipeline, './spaceship-titanic-kfp-sona.yaml')


# pipeline.py run >> pipeline.yaml create
# kubeflow dashboard > pipeline update