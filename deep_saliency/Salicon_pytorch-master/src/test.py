import numpy as np
import torch
import torch.nn as nn
import cv2

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"..")))

import argparse

# from src.data_utils import getTest_loader
from src.data_utils_changed import getTest_loader
from src.salicon_model import Salicon



def test(model,device,test_loader,output_dir='results',model_save_name=''):
    model.eval()

    with torch.no_grad():
        for input,target,name in test_loader:
            fine_img,coarse_img=input
            fine_img=fine_img.unsqueeze(0).to(device)
            coarse_img=coarse_img.unsqueeze(0).to(device)

            pred=model(fine_img,coarse_img)

            pred_image=pred.squeeze()
            h,w=target.shape[-2],target.shape[-1]

            smap=(pred_image-torch.min(pred_image))/((torch.max(pred_image)-torch.min(pred_image)))
            smap=smap.cpu().numpy()
            smap=cv2.resize(smap,(w,h),interpolation=cv2.INTER_CUBIC)
            # smap=cv2.GaussianBlur(smap,(75,75),25,cv2.BORDER_DEFAULT)
            smap=cv2.GaussianBlur(smap,(11,11),11,cv2.BORDER_DEFAULT)

            print(test_loader.fine_coarse_add)
            # cv2.namedWindow('smap',cv2.WINDOW_NORMAL)
            # cv2.imshow('smap',smap)
            smap = (smap*255).astype(np.uint8)
            cv2.imwrite(os.path.join(os.path.abspath('..'),output_dir,'saliency',name), smap)
            # print(target.shape)
            print(target.shape)
            print(smap.shape)
            print(torch.max(target))
            target = target.squeeze().cpu().numpy()
            target = (target*255).astype(np.uint8)
            target = cv2.resize(target,(w,h),interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(os.path.abspath('..'),output_dir,'target',name), target)
            # coarse_img = coarse_img.squeeze().cpu().numpy()
            # print(coarse_img.shape)
            # coarse_img = cv2.resize(coarse_img,(w,h),interpolation=cv2.INTER_CUBIC)
            # print(coarse_img.shape)
            # cv2.imwrite(os.path.join(os.path.abspath('..'),'results','input',name), coarse_img)

            # target=target.squeeze().cpu().numpy()
            # cv2.namedWindow('target',cv2.WINDOW_NORMAL)
            # cv2.imshow('target',target)

            # cv2.waitKey(0)

def main():
    parser=argparse.ArgumentParser()
    np.random.seed(12)

    # dataset type
    parser.add_argument('--test_dataset',type=str,default='osie')
    parser.add_argument('--test_dataset_dir',type=str,default='/osie_dataset/data')
    parser.add_argument('--test_img_dir', type=str, default='stimuli')
    parser.add_argument('--test_label_dir', type=str, default='fixation_maps')
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='')

    # gpu
    parser.add_argument('--gpu',default=True,action='store_true')

    # model dir
    parser.add_argument('--model_dir', type=str, default='src/salicon_model.pth')

    args=parser.parse_args()

    # Get dataloader (test)
    # if args.test_dataset=='mit1003':
    #     args.test_dataset_dir='/mit1003_dataset'
    #     args.test_img_dir='ALLSTIMULI'
    #     args.test_label_dir='ALLFIXATIONMAPS'
    # elif args.test_dataset=='osie':
    #     args.test_dataset_dir='/osie_dataset/data'
    #     args.test_img_dir='stimuli'
    #     args.test_label_dir='fixation_maps'
    # else:
    #     raise NotImplemented

    # test_dataset_dir=os.path.abspath('..')+args.test_dataset_dir
    test_dataset_dir=args.test_dataset_dir
    dataloaders=getTest_loader(test_dataset_dir,args.test_img_dir,args.test_label_dir)

    # init the model
    model_weight=os.path.join(os.path.abspath('..'),args.model_dir)

    device=torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    model_trained=Salicon()
    model_trained.load_state_dict(torch.load(model_weight))
    model_trained.to(device)

    print("Begin test, Device: {}".format(device))

    if not os.path.exists(os.path.join(os.path.abspath('..'),args.output_dir,'target')):
        os.makedirs(os.path.join(os.path.abspath('..'),args.output_dir,'target'))
    if not os.path.exists(os.path.join(os.path.abspath('..'),args.output_dir,'input')):
        os.makedirs(os.path.join(os.path.abspath('..'),args.output_dir,'input'))
    if not os.path.exists(os.path.join(os.path.abspath('..'),args.output_dir,'saliency')):
        os.makedirs(os.path.join(os.path.abspath('..'),args.output_dir,'saliency'))

    # test the model
    test(model_trained,device,dataloaders,model_save_name=args.model_name,output_dir=args.output_dir)

if __name__ == '__main__':
    main()