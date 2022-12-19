from __future__ import division
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os, cv2, sys
import numpy as np
from config_ar import *
from utilities import preprocess_images, preprocess_maps, postprocess_predictions
from model import ml_net_model, loss


def generator(b_s, phase_gen='train', train_data_csv=None, test_data_csv=None):
    if phase_gen == 'train':
        images = [imgs_train_path + f for f in os.listdir(imgs_train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        maps = [maps_train_path + f for f in os.listdir(maps_train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if train_data_csv is not None:
            with open(os.path.join(train_data_path, train_data_csv), "r") as f:
                image_paths = f.read().splitlines()
            image_names = []
            images = []
            maps = []
            cnt = 0
            for image_path in image_paths:
                cnt = cnt+1
                if cnt == 1:
                    continue
                name = image_path.split(',')
                image_names.append(name[2])    # superimposed images
                images.append(os.path.join(imgs_train_path,name[2]))    # superimposed images
                maps.append(os.path.join(maps_train_path,name[2]))    # superimposed images
        print(len(images))
        print(len(maps))
    elif phase_gen == 'val':
        images = [imgs_train_path + f for f in os.listdir(imgs_train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        maps = [maps_train_path + f for f in os.listdir(maps_train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if test_data_csv is not None:
            with open(os.path.join(test_data_path, test_data_csv), "r") as f:
                image_paths = f.read().splitlines()
            image_names = []
            images = []
            maps = []
            cnt = 0
            for image_path in image_paths:
                cnt = cnt+1
                if cnt == 1:
                    continue
                name = image_path.split(',')
                image_names.append(name[2])    # superimposed images
                images.append(os.path.join(imgs_val_path,name[2]))    # superimposed images
                maps.append(os.path.join(maps_val_path,name[2]))    # map images
        print(len(images))
        print(len(maps))
    else:
        raise NotImplementedError

    images.sort()
    maps.sort()

    counter = 0
    while True:
        yield preprocess_images(images[counter:counter + b_s], shape_r, shape_c), preprocess_maps(maps[counter:counter + b_s], shape_r_gt, shape_c_gt)
        counter = (counter + b_s) % len(images)


def generator_test(b_s, file_names):
    # images = [imgs_test_path + f for f in os.listdir(imgs_test_path) if f.endswith('.jpg')]
    # images.sort()
    images = file_names

    counter = 0
    while True:
        yield preprocess_images(images[counter:counter + b_s], shape_r, shape_c)
        counter = (counter + b_s) % len(images)


if __name__ == '__main__':
    phase = sys.argv[1]

    model = ml_net_model(img_cols=shape_c, img_rows=shape_r, downsampling_factor_product=10)
    sgd = SGD(lr=1e-3, decay=0.0005, momentum=0.9, nesterov=True)
    print("Compile ML-Net Model")
    model.compile(sgd, loss)

    if phase == 'train':
        model_dir = sys.argv[2]
        train_data_csv = sys.argv[3]
        test_data_csv = sys.argv[4]
        if not os.path.exists('weights/'):
            os.makedirs('weights/')

        print("Training ML-Net")
        model.load_weights('weights/'+'salicon_weights.pkl')
        model.fit_generator(generator(b_s=b_s, train_data_csv=train_data_csv, test_data_csv=test_data_csv), nb_imgs_train, nb_epoch=nb_epoch,
                            validation_data=generator(b_s=b_s, phase_gen='val', train_data_csv=train_data_csv, test_data_csv=test_data_csv), nb_val_samples=nb_imgs_val,
                            callbacks=[EarlyStopping(patience=5),
                                       ModelCheckpoint('weights/'+model_dir+'_weights.pkl', save_best_only=True)])
                                    #    ModelCheckpoint('weights.mlnet.{epoch:02d}-{val_loss:.4f}.pkl', save_best_only=True)])

    elif phase == "test":
        # path of output folder
        output_folder = ''

        if len(sys.argv) < 2:
            raise SyntaxError
        imgs_test_path = sys.argv[2]
        model_dir = sys.argv[3]
        output_folder = sys.argv[4]
        test_data_csv = sys.argv[5]

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        file_names = [f for f in os.listdir(imgs_test_path) if f.endswith('.jpg')]
        if test_data_csv is not None:
            with open(os.path.join(test_data_path, test_data_csv), "r") as f:
                image_paths = f.read().splitlines()
            image_names = []
            file_names = []
            cnt = 0
            for image_path in image_paths:
                cnt = cnt+1
                if cnt == 1:
                    continue
                name = image_path.split(',')
                image_names.append(name[2])    # superimposed images
                file_names.append(os.path.join(imgs_test_path,name[2]))    # superimposed images
        file_names.sort()
        nb_imgs_test = len(file_names)

        print("Load weights ML-Net")
        model.load_weights('weights/'+model_dir+'_weights.pkl')
        # model.load_weights('mlnet_salicon_weights.pkl')

        print("Predict saliency maps for " + imgs_test_path)
        predictions = model.predict_generator(generator_test(b_s=1, file_names=file_names), nb_imgs_test)

        for pred, name in zip(predictions, file_names):
            print(name)
            original_image = cv2.imread(name, 0)
            res = postprocess_predictions(pred[0], original_image.shape[0], original_image.shape[1])
            cv2.imwrite(output_folder + '%s' % os.path.basename(name), res.astype(int))

    else:
        raise NotImplementedError
