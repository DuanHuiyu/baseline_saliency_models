#########################################################################
# MODEL PARAMETERS														#
#########################################################################
# version (0 for SAM-VGG and 1 for SAM-ResNet)
version = 1
# batch size
b_s = 1
# number of rows of input images
shape_r = 240
# number of cols of input images
shape_c = 320
# number of rows of downsampled maps
shape_r_gt = 30
# number of cols of downsampled maps
shape_c_gt = 40
# number of rows of model outputs
shape_r_out = 480
# number of cols of model outputs
shape_c_out = 640
# final upsampling factor
upsampling_factor = 16
# number of epochs
nb_epoch = 10
# number of timestep
nb_timestep = 4
# number of learned priors
nb_gaussian = 16

#########################################################################
# TRAINING SETTINGS										            	#
#########################################################################
# path of training images
imgs_train_path = '/data/duan/sal_all/datasets/ar_saliency_database/small_size/Superimposed/'
# path of training maps
maps_train_path = '/data/duan/sal_all/datasets/ar_saliency_database/small_size/fixMaps/'
# path of training fixation maps
fixs_train_path = '/data/duan/sal_all/datasets/ar_saliency_database/small_size/fixPts/'
# number of training images
nb_imgs_train = 1080
# path of validation images
imgs_val_path = '/data/duan/sal_all/datasets/ar_saliency_database/small_size/Superimposed/'
# path of validation maps
maps_val_path = '/data/duan/sal_all/datasets/ar_saliency_database/small_size/fixMaps/'
# path of validation fixation maps
fixs_val_path = '/data/duan/sal_all/datasets/ar_saliency_database/small_size/fixPts/'
# number of validation images
nb_imgs_val = 270
# train data csv
train_data_path = '/data/duan/sal_all/datasets/ar_saliency_database/'
# test data csv
test_data_path = '/data/duan/sal_all/datasets/ar_saliency_database/'