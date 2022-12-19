import math

#########################################################################
# MODEL PARAMETERS														#
#########################################################################
# batch size
b_s = 10
# number of rows of input images
shape_r = 480
# number of cols of input images
shape_c = 640
# number of rows of predicted maps
shape_r_gt = int(math.ceil(shape_r / 8))
# number of cols of predicted maps
shape_c_gt = int(math.ceil(shape_c / 8))
# number of epochs
nb_epoch = 20

#########################################################################
# TRAINING SETTINGS										            	#
#########################################################################
# path of training images
imgs_train_path = '/data/duan/sal_all/datasets/ar_saliency_database/small_size/Superimposed'
# path of training maps
maps_train_path = '/data/duan/sal_all/datasets/ar_saliency_database/small_size/fixMaps'
# number of training images
nb_imgs_train = 1080
# path of validation images
imgs_val_path = '/data/duan/sal_all/datasets/ar_saliency_database/small_size/Superimposed'
# path of validation maps
maps_val_path = '/data/duan/sal_all/datasets/ar_saliency_database/small_size/fixMaps'
# number of validation images
nb_imgs_val = 270
# train data csv
train_data_path = '/data/duan/sal_all/datasets/ar_saliency_database/'
# test data csv
test_data_path = '/data/duan/sal_all/datasets/ar_saliency_database/'