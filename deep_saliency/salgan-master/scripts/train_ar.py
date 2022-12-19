#   Two mode of training available:
#       - BCE: CNN training, NOT Adversarial Training here. Only learns the generator network.
#       - SALGAN: Adversarial Training. Updates weights for both Generator and Discriminator.
#   The training used data previously  processed using "01-data_preocessing.py"
import os
import numpy as np
import sys
# import cPickle as pickle
import pickle
import random
import cv2
import theano
import theano.tensor as T
import lasagne

from tqdm import tqdm
from constants import *
from models.model_salgan import ModelSALGAN
from models.model_bce import ModelBCE
from utils import *

import glob

flag = str(sys.argv[1])
train_img_path = str(sys.argv[2])
train_sal_path = str(sys.argv[3])
val_img_path = str(sys.argv[4])
val_sal_path = str(sys.argv[5])
model_name = str(sys.argv[6])

train_data_csv = str(sys.argv[7])
test_data_csv = str(sys.argv[8])

INPUT_SIZE = (256, 192)

train_imgs = []
if train_data_csv is not None:
    with open(train_data_csv, "r") as f:
        image_paths = f.read().splitlines()
    train_imgs = []
    cnt = 0
    for image_path in image_paths:
        cnt = cnt+1
        if cnt == 1:
            continue
        name = image_path.split(',')
        train_imgs.append(name[2])    # superimposed images

if test_data_csv is not None:
    with open(test_data_csv, "r") as f:
        image_paths = f.read().splitlines()
    val_imgs = []
    val_image_paths = []
    cnt = 0
    for image_path in image_paths:
        cnt = cnt+1
        if cnt == 1:
            continue
        name = image_path.split(',')
        val_imgs.append(name[2])    # superimposed images
        val_image_paths.append(os.path.join(val_img_path, name[2]))

if not os.path.exists(DIR_TO_SAVE):
    os.makedirs(DIR_TO_SAVE)


def img_data_process(img_path):
    img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    imageResized = cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_AREA)
    return imageResized

def sal_data_process(sal_path):
    sal = cv2.cvtColor(cv2.imread(sal_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
    saliencyResized = cv2.resize(sal, INPUT_SIZE, interpolation=cv2.INTER_AREA)
    return saliencyResized

def bce_batch_iterator(model, train_data, validation_sample):
    num_epochs = 10
    n_updates = 1
    nr_batches_train = int(len(train_data) / model.batch_size)

    with np.load(DIR_TO_SAVE + '/gen_modelWeights'+'_salicon_bce50'+'.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(model.net['output'], param_values)

    for current_epoch in tqdm(range(num_epochs), ncols=20):
        e_cost = 0.

        random.shuffle(train_data)

        for currChunk in chunks(train_data, model.batch_size):

            if len(currChunk) != model.batch_size:
                continue

            batch_input = np.asarray([img_data_process(os.path.join(train_img_path,x)).astype(theano.config.floatX).transpose(2, 0, 1) for x in currChunk],
                                     dtype=theano.config.floatX)

            batch_output = np.asarray([sal_data_process(os.path.join(train_sal_path,y.replace('.jpg','.png'))).astype(theano.config.floatX) / 255. for y in currChunk],
                                      dtype=theano.config.floatX)
            batch_output = np.expand_dims(batch_output, axis=1)

            # train generator with one batch and discriminator with next batch
            G_cost = model.G_trainFunction(batch_input, batch_output)
            e_cost += G_cost
            n_updates += 1

        e_cost /= nr_batches_train

        print('Epoch:', current_epoch, ' train_loss->', e_cost)

        if current_epoch % 5 == 0:
            num_random = random.choice(range(len(val_image_paths)))
            validation_sample = img_data_process(val_image_paths[num_random])

            np.savez(DIR_TO_SAVE + '/gen_modelWeights'+model_name+'.npz',
                     *lasagne.layers.get_all_param_values(model.net['output']))
            predict(model=model, image_stimuli=validation_sample, num_epoch=current_epoch, path_output_maps=DIR_TO_SAVE)


def salgan_batch_iterator(model, train_data, validation_sample):
    num_epochs = 10
    nr_batches_train = int(len(train_data) / model.batch_size)
    n_updates = 1

    with np.load(DIR_TO_SAVE + '/gen_modelWeights'+'_salicon_salgan'+'.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(model.net['output'], param_values)

    for current_epoch in tqdm(range(num_epochs), ncols=20):

        g_cost = 0.
        d_cost = 0.
        e_cost = 0.

        random.shuffle(train_data)

        for currChunk in chunks(train_data, model.batch_size):

            if len(currChunk) != model.batch_size:
                continue

            batch_input = np.asarray([img_data_process(os.path.join(train_img_path,x)).astype(theano.config.floatX).transpose(2, 0, 1) for x in currChunk],
                                     dtype=theano.config.floatX)

            batch_output = np.asarray([sal_data_process(os.path.join(train_sal_path,y.replace('.jpg','.png'))).astype(theano.config.floatX) / 255. for y in currChunk],
                                      dtype=theano.config.floatX)
            batch_output = np.expand_dims(batch_output, axis=1)

            # train generator with one batch and discriminator with next batch
            if n_updates % 2 == 0:
                G_obj, D_obj, G_cost = model.G_trainFunction(batch_input, batch_output)
                d_cost += D_obj
                g_cost += G_obj
                e_cost += G_cost
            else:
                G_obj, D_obj, G_cost = model.D_trainFunction(batch_input, batch_output)
                d_cost += D_obj
                g_cost += G_obj
                e_cost += G_cost

            n_updates += 1

        g_cost /= nr_batches_train
        d_cost /= nr_batches_train
        e_cost /= nr_batches_train

        # Save weights every 3 epoch
        if current_epoch % 3 == 0:
            num_random = random.choice(range(len(val_image_paths)))
            validation_sample = img_data_process(val_image_paths[num_random])
            
            np.savez(DIR_TO_SAVE + '/gen_modelWeights'+model_name+'.npz',
                     *lasagne.layers.get_all_param_values(model.net['output']))
            np.savez(DIR_TO_SAVE + '/disrim_modelWeights'+model_name+'.npz',
                     *lasagne.layers.get_all_param_values(model.discriminator['prob']))
            predict(model=model, image_stimuli=validation_sample, num_epoch=current_epoch, path_output_maps=DIR_TO_SAVE)
        print('Epoch:', current_epoch, ' train_loss->', (g_cost, d_cost, e_cost))


def train():
    """
    Train both generator and discriminator
    :return:
    """
    # # Load data
    # print 'Loading training data...'
    # with open('../saliency-2016-lsun/validationSample240x320.pkl', 'rb') as f:
    # # with open(TRAIN_DATA_DIR, 'rb') as f:
    #     train_data = pickle.load(f)
    # print '-->done!'

    # print 'Loading validation data...'
    # with open('../saliency-2016-lsun/validationSample240x320.pkl', 'rb') as f:
    # # with open(VALIDATION_DATA_DIR, 'rb') as f:
    #     validation_data = pickle.load(f)
    # print '-->done!'

    # # Choose a random sample to monitor the training
    # num_random = random.choice(range(len(validation_data)))
    # validation_sample = validation_data[num_random]
    # cv2.imwrite('./' + DIR_TO_SAVE + '/validationRandomSaliencyGT.png', validation_sample.saliency.data)
    # cv2.imwrite('./' + DIR_TO_SAVE + '/validationRandomImage.png', cv2.cvtColor(validation_sample.image.data,
    #                                                                             cv2.COLOR_RGB2BGR))


    # Create network

    if flag == 'salgan':
        model = ModelSALGAN(INPUT_SIZE[0], INPUT_SIZE[1])
        # Load a pre-trained model
        # load_weights(net=model.net['output'], path="nss/gen_", epochtoload=15)
        # load_weights(net=model.discriminator['prob'], path="test_dialted/disrim_", epochtoload=54)
        salgan_batch_iterator(model, train_imgs, val_imgs)

    elif flag == 'bce':
        model = ModelBCE(INPUT_SIZE[0], INPUT_SIZE[1])
        # Load a pre-trained model
        # load_weights(net=model.net['output'], path='test/gen_', epochtoload=15)
        bce_batch_iterator(model, train_imgs, val_imgs)
    else:
        print("Invalid input argument.")
if __name__ == "__main__":
    train()
