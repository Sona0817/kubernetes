import argparse
import os
import shutil
from bento_service import Spaceship_Service
from load_model import load

def bento_serve(opt):
    surface_classifier_service = Spaceship_Service()
    model = load(model_name=opt.model_name, version=opt.version)

    # Pack the newly trained model artifact
    surface_classifier_service.pack('rf', model)

    # Save the prediction service to disk for model bento
    saved_path = surface_classifier_service.save()

    os.makedirs(os.path.join(opt.data_path, "bento"), exist_ok=True)
    shutil.move(saved_path, os.path.join(opt.data_path, "bento"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, help='dataset root path')
    parser.add_argument('--model-name', type=str, help='MLflow model name')
    parser.add_argument('--version', type=int, help='MLFlow model version')
    opt = parser.parse_args()

    bento_serve(opt)