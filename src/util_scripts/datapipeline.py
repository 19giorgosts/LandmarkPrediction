"""

Construct a data pipeline using the MONAI API

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""

import os
from monai.transforms import *
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
import joblib
import torch
from monai.transforms import StdShiftIntensity
import numpy as np
import cv2
# from util_scripts.utils import *


# Data dicts for synthetic transforms (eg: training/evaluation w.r.t repeatability)
def create_data_dicts_dl_reg(patient_dir_list=None):

    data_dicts = []
    # print(patient_dir_list)
    for p_dir in patient_dir_list:
        # p_id = p_dir.split(os.sep)[-1]
        p_id = p_dir.split('_')[1]
        # print(p_id)
        # scan_dirs  = [f.path for f in os.scandir(p_dir) if f.is_dir()]
        # print(scan_dirs)
        # for s_dir in scan_dirs:
        #     s_id = s_dir.split(os.sep)[-1]
        #     print(s_id)
        data_dict = {}
        data_dict['image'] = os.path.join(p_dir, 'mr_pv_res.nii')
        data_dict['images_hat'] = os.path.join(p_dir, 'images_hat.nii')
        data_dict['prostate_mask'] = os.path.join(p_dir, 'prostate_mask.nii')

        data_dict['patient_id'] = p_id
        # data_dict['scan_id'] = s_id
        data_dicts.append(data_dict)
        # print(data_dict.keys())

    return data_dicts

# Data dicts for "real" paired data
def create_data_dicts_lesion_matching_inference(patient_dir_list=None):

    data_dicts = []

    for p_dir in patient_dir_list:
        p_id = p_dir.split(os.sep)[-1]
        scan_dirs  = [f.path for f in os.scandir(p_dir) if f.is_dir()]
        data_dict = {}
        data_dict['patient_id'] = p_id
        for idx, s_dir in enumerate(scan_dirs):
            s_id = s_dir.split(os.sep)[-1]
            data_dict['image_{}'.format(idx+1)] = os.path.join(s_dir, 'DCE_vessel_image.nii')
            data_dict['liver_mask_{}'.format(idx+1)] = os.path.join(s_dir, 'LiverMask.nii')
            # data_dict['vessel_mask_{}'.format(idx+1)] = os.path.join(s_dir, 'vessel_mask.nii')
            # if os.path.exists(os.path.join(s_dir, 'vessel_mask.nii')) is False:
            #     print('Vessel mask does not exist for Patient {}, scan-ID : {}'.format(p_id, s_dir))
            #     data_dict['vessel_mask_{}'.format(idx+1)] = os.path.join(s_dir, 'LiverMask.nii')
            # data_dict['scan_id_{}'.format(idx)] = s_id

        data_dicts.append(data_dict)

    return data_dicts



def create_dataloader_dl_reg(data_dicts=None, train=True, batch_size=4, num_workers=4, data_aug=True, patch_size=(96,96,48),shuffle=False):

    if train is True:
        if data_aug is True:
            transforms = Compose([LoadImaged(keys=["image", "images_hat", "prostate_mask"]),

                                  # Add fake channel to the prostate_mask
                                  AddChanneld(keys=["image", "images_hat", "prostate_mask"]),

                                  Orientationd(keys=["image", "images_hat", "prostate_mask"], axcodes="RAS"),

                                  ### Isotropic spacing
                                  Spacingd(keys=["image", "images_hat", "prostate_mask"],
                                           pixdim=(2, 2, 2),
                                           mode=("bilinear", "bilinear", "nearest")),

                                  # Extract 3-D patches
                                  # RandRotated(keys=["image", "prostate_mask"],
                                  #             range_x=(np.pi/180)*15,
                                  #             range_y=(np.pi/180)*15,
                                  #             range_z=(np.pi/180)*15,
                                  #             mode=["bilinear", "nearest"],
                                  #             prob=0.5),

                                  # RandAxisFlipd(keys=["image", "prostate_mask"],
                                  #              prob=0.7),
                                  #
                                  # RandZoomd(keys=["image", "prostate_mask"],
                                  #           p=0.1),

                                  # RandAffined(
                                  #       keys=["image"],
                                  #       mode=("bilinear"),
                                  #       prob=1.0,
                                  #       translate_range=(20, 20, 0),
                                  #   ),

                                  # ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),

                                  NormalizeIntensityd(keys=["image", "images_hat"],
                                                      nonzero=True,
                                                      channel_wise=True),

                                  # ScaleIntensityd(keys=["image","images_hat"], minv=0.0, maxv=1.0),

                                  RandCropByPosNegLabeld(keys=["image", "images_hat", "prostate_mask"],
                                                         label_key="prostate_mask",
                                                         spatial_size=patch_size,
                                                         pos=1.0,
                                                         neg=0.0),

                                  EnsureTyped(keys=["image", "images_hat", "prostate_mask"])
                                  ])
        else:
            ## no rotations, scaling and axis flipping for validation set
            transforms = Compose([LoadImaged(keys=["image", "images_hat", "prostate_mask"]),

                                  # Add fake channel to the prostate_mask
                                  AddChanneld(keys=["image", "images_hat", "prostate_mask"]),

                                  Orientationd(keys=["image", "images_hat","prostate_mask"], axcodes="RAS"),

                                  # Isotropic spacing
                                  Spacingd(keys=["image", "images_hat", "prostate_mask"],
                                           pixdim=(2, 2, 2),
                                           mode=("bilinear", "bilinear", "nearest")),

                                  # Extract 3-D patches
                                  NormalizeIntensityd(keys=["image", "images_hat"],
                                                      nonzero=True,
                                                      channel_wise=True),

                                  # ScaleIntensityd(keys=["image","images_hat"], minv=0.0, maxv=1.0),

                                  RandCropByPosNegLabeld(keys=["image", "images_hat", "prostate_mask"],
                                                         label_key="prostate_mask",
                                                         spatial_size=patch_size,
                                                         pos=1.0,
                                                         neg=0.0),
                                  EnsureTyped(keys=["image", "images_hat", "prostate_mask"])
                                  ])

    else:
        transforms = Compose([LoadImaged(keys=["image", "prostate_mask"]),

                              AddChanneld(keys=["image", "prostate_mask"]),

                              Orientationd(keys=["image", "prostate_mask"], axcodes="RAS"),

                              Spacingd(keys=["image", "prostate_mask"],
                                       pixdim=(2, 2),
                                       mode=("bilinear", "nearest")),

                              NormalizeIntensityd(keys=["image"],
                                                  nonzero=True,
                                                  channel_wise=True),

                              # ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),

                              EnsureTyped(keys=["image", "prostate_mask"])
                              ])

    ds = CacheDataset(data=data_dicts,
                      transform=transforms,
                      cache_rate=1.0,
                      num_workers=num_workers)

    # only shuffling training set
    loader = DataLoader(ds,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers)

    return loader, transforms


# With "real" paired data
def create_dataloader_lesion_matching_inference(data_dicts=None, batch_size=4, num_workers=4):

    transforms = Compose([LoadImaged(keys=["image_1",  "liver_mask_1", "vessel_mask_1",
                                           "image_2",  "liver_mask_2", "vessel_mask_2"]),

                          # Add fake channel to the liver_mask
                          AddChanneld(keys=["image_1", "liver_mask_1", "vessel_mask_1",
                                            "image_2",  "liver_mask_2", "vessel_mask_2"]),

                          Orientationd(keys=["image_1", "liver_mask_1", "vessel_mask_1",
                                             "image_2",  "liver_mask_2", "vessel_mask_2"], axcodes="RAS"),

                          # Isotropic spacing
                          Spacingd(keys=["image_1", "liver_mask_1", "vessel_mask_1",
                                         "image_2",  "liver_mask_2", "vessel_mask_2"],
                                   pixdim=(1.543, 1.543, 1.543),
                                   mode=("bilinear", "nearest", "nearest",
                                         "bilinear", "nearest", "nearest")),

                          # Extract 128x128x64 3-D patches
                          RandCropByPosNegLabeld(keys=["image_1", "liver_mask_1", "vessel_mask_1",
                                                       "image_2", "liver_mask_2", "vessel_mask_2"],
                                                 label_key="liver_mask_1",
                                                 spatial_size=(128, 128, 64),
                                                 pos=1.0,
                                                 neg=0.0),

                          NormalizeIntensityd(keys=["image_1", "image_2"],
                                              nonzero=True,
                                              channel_wise=True),


                          EnsureTyped(keys=["image_1", "liver_mask_1", "vessel_mask_1",
                                            "image_2", "liver_mask_2", "vessel_mask_2"])
                          ])


    ds = CacheDataset(data=data_dicts,
                      transform=transforms,
                      cache_rate=1.0,
                      num_workers=num_workers)

    loader = DataLoader(ds,
                        batch_size=batch_size,
                        shuffle=train,
                        num_workers=num_workers)

    return loader, transforms

def create_dataloader_liver_seg(data_dicts=None, train=True, batch_size=4):

    if train is True:
        transforms = Compose([LoadImaged(keys=["image", "label"]),

                              # Add fake channel to the label
                              AddChanneld(keys=["label"]),

                              # Make sure image is channel first
                              EnsureChannelFirstd(keys=["image"]),

                              Orientationd(keys=["image", "label"], axcodes="RAS"),

                              # Isotropic spacing
                              Spacingd(keys=["image", "label"],
                                       pixdim=(1.543, 1.543, 1.543),
                                       mode=("bilinear", "nearest")),

                              # Extract 128x128x48 3-D patches
                              RandSpatialCropd(keys=["image", "label"],
                                               roi_size=[128, 128, 48],
                                               random_size=False),

                              RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),

                              RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),

                              RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),

                              NormalizeIntensityd(keys=["image"],
                                                  nonzero=True,
                                                  channel_wise=True),

                              RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),

                              RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),

                              EnsureTyped(keys=["image", "label"])
                              ])

    else:
        transforms = Compose([LoadImaged(keys=["image", "label"]),

                              # Add fake channel to the label
                              AddChanneld(keys=["label"]),

                              EnsureChannelFirstd(keys=["image"]),

                              Orientationd(keys=["image", "label"], axcodes="RAS"),

                              Spacingd(keys=["image", "label"],
                                       pixdim=(1.543, 1.543, 1.543),
                                       mode=("bilinear", "nearest")),

                              NormalizeIntensityd(keys=["image"],
                                                  nonzero=True,
                                                  channel_wise=True),

                              EnsureTyped(keys=["image", "label"])
                              ])

    ds = CacheDataset(data=data_dicts,
                      transform=transforms,
                      cache_rate=1.0,
                      num_workers=4)

    loader = DataLoader(ds,
                        batch_size=batch_size,
                        shuffle=train,
                        num_workers=4)

    if train is True:
        return loader
    else:
        return loader, transforms

def debug_dataloader_liver_seg(data_dicts=None, batch_size=4):
    """
    Create a bare-bones transform pipeline to flush out bugs in the pipeline

    """

    transforms = Compose([LoadImaged(keys=["image", "label"]),

                          # Add fake channel to the label
                          AddChanneld(keys=["label"]),

                          EnsureChannelFirstd(keys=["image"]),

                          Orientationd(keys=["image", "label"], axcodes="RAS"),

                          Spacingd(keys=["image", "label"],
                                   pixdim=(1.543, 1.543, 1.543),
                                   mode=("bilinear", "nearest")),

                          NormalizeIntensityd(keys=["image"],
                                              nonzero=True,
                                              channel_wise=True),

                          RandSpatialCropd(keys=["image", "label"],
                                           roi_size=[128, 128, 48],
                                           random_size=False),

                          EnsureTyped(keys=["image", "label"])
                          ])


    print('Instanced dataset class')

    ds = Dataset(data=data_dicts,
                 transform=transforms)

    loader = DataLoader(ds,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=4)

    print('Created data loader')
    return loader


def shift_intensity(images):

    assert(isinstance(images, torch.Tensor))
    print(images.shape)
    factor = np.random.uniform(low=0.6, high=1.0)
    images_shifted = StdShiftIntensity(factor=factor)(images)
    return images_shifted

if __name__ == '__main__':

    train_patients = joblib.load('train_patients.pkl')
    train_dicts = create_data_dicts_liver_seg(train_patients)

    loader = debug_dataloader_liver_seg(train_dicts, batch_size=4)

    for batch_data in loader:
        images = batch_data['image']
        print(images.shape)
