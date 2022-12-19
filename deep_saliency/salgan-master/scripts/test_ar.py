import os
import numpy as np
from tqdm import tqdm
import cv2
import glob
from utils import *
from constants import *
from models.model_bce import ModelBCE
import sys

def my_load_weights(net, model_name):
    """
    Load a pretrained model
    :param epochtoload: epoch to load
    :param net: model object
    :param path: path of the weights to be set
    """
    with np.load(DIR_TO_SAVE + '/gen_modelWeights'+model_name+'.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(net, param_values)


def img_data_process(img_path):
    img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    imageResized = cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_AREA)
    return imageResized


def test(path_to_images, path_output_maps, model_to_test=None, test_data_csv=None):
    # list_img_files = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(path_to_images, '*'))]
    list_img_files = glob.glob(os.path.join(path_to_images, '*.png'))
    list_img_files += glob.glob(os.path.join(path_to_images, '*.jpg'))
    if test_data_csv is not None:
        with open(test_data_csv, "r") as f:
            image_paths = f.read().splitlines()
        val_imgs = []
        list_img_files = []
        cnt = 0
        for image_path in image_paths:
            cnt = cnt+1
            if cnt == 1:
                continue
            name = image_path.split(',')
            val_imgs.append(name[2])    # superimposed images
            list_img_files.append(os.path.join(path_to_images, name[2]))
    # Load Data
    # list_img_files.sort()
    for curr_file in tqdm(list_img_files, ncols=20):
        print(curr_file)
        # img = cv2.cvtColor(cv2.imread(os.path.join(path_to_images, curr_file + '.jpg'), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img_name = os.path.basename(curr_file).split('.')[0]
        img = img_data_process(curr_file)
        predict(model=model_to_test, image_stimuli=img, name=img_name, path_output_maps=path_output_maps)


def main():
    img_path = str(sys.argv[1])
    model_name = str(sys.argv[2])
    output_path = str(sys.argv[3])
    test_data_csv = str(sys.argv[4])

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Create network
    model = ModelBCE(INPUT_SIZE[0], INPUT_SIZE[1], batch_size=8)
    # Here need to specify the epoch of model sanpshot
    my_load_weights(model.net['output'], model_name)
    # Here need to specify the path to images and output path
    test(path_to_images=img_path, path_output_maps=output_path, model_to_test=model, test_data_csv=test_data_csv)

if __name__ == "__main__":
    main()
