
import torch
from torch.utils.data import DataLoader
import os
import sys
sys.path.append(os.path.join(os.path.expanduser('E:/'), 'LandmarkBasedRegistration', 'src', 'util_scripts/'))
sys.path.append(os.path.join(os.path.expanduser('E:/'), 'LandmarkBasedRegistration', 'src', 'arch/'))

import wandb
import torch.optim as optim
from arch.model import LesionMatchingModel
import math
from arch.loss import *
from util_scripts.datapipeline import *
from util_scripts.deformations import *
from tqdm import tqdm
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torchsummary import summary

def build_optimizer(network, optimizer, learning_rate, L2reg):
    if optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=learning_rate,
                               weight_decay=L2reg)
    return optimizer


def build_transforms(pz):
    transforms = Compose([LoadImaged(keys=["image", "prostate_mask"]),

                          # Add fake channel to the prostate_mask
                          AddChanneld(keys=["image", "prostate_mask"]),

                          Orientationd(keys=["image", "prostate_mask"], axcodes="RAS"),

                          # # Isotropic spacing
                          # Spacingd(keys=["image", "prostate_mask"],
                          #          pixdim=(2, 2, 2),
                          #          mode=("bilinear", "nearest")),

                          # Extract 3-D patches
                          RandCropByPosNegLabeld(keys=["image", "prostate_mask"],
                                                 label_key="prostate_mask",
                                                 spatial_size=pz,
                                                 pos=1.0,
                                                 neg=0.0),

                          # RandRotated(keys=["image", "prostate_mask"],
                          #             range_x=(np.pi / 180) * 30,
                          #             range_y=(np.pi / 180) * 15,
                          #             range_z=(np.pi / 180) * 15,
                          #             mode=["bilinear", "nearest"],
                          #             prob=0.5),

                          # RandAxisFlipd(keys=["image", "prostate_mask"],
                          #              prob=0.7),

                          # RandZoomd(keys=["image", "prostate_mask"],
                          #           p=0.3),

                          NormalizeIntensityd(keys=["image"],
                                              nonzero=True),

                          EnsureTyped(keys=["image", "prostate_mask"])
                          ])
    return transforms

def build_dataset(batch_size,pz):
    ## select a subset of 20 patients for hyperparameter tuning
    val_patients = joblib.load('val_patients.pkl')[0:20]
    # print(train_patients[0:21])
    val_dicts = create_data_dicts_dl_reg(val_patients)

    ds = CacheDataset(data=val_dicts,
                      transform=build_transforms(pz),
                      cache_rate=1.0,
                      num_workers=4
                      )


    loader = DataLoader(ds, batch_size=batch_size)                                   # Here are then fed to the network with a defined batch size

    return loader

def build_network(K,W):
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2'  # Multi-gpu selector for training
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = LesionMatchingModel(K, W).to(device)
    return net


def train_epoch(network, loader, optimizer,keypoints):
    cumu_loss = 0
    # cost = RmseLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    nbatches = len(loader)
    pbar = tqdm(enumerate(loader), desc="training", total=nbatches, unit="batches")
    for batch_idx, batch_data in pbar:
        images, prostate_mask = (batch_data['image'], batch_data['prostate_mask'])

        batch_deformation_grid = create_batch_deformation_grid(shape=images.shape,
                                                               device=images.device,
                                                               dummy=False)
        images_hat = F.grid_sample(input=images,
                                   grid=batch_deformation_grid,
                                   align_corners=True,
                                   mode="bilinear")

        optimizer.zero_grad()
        outputs = network(images.to(device),images_hat.to(device),training=True)
        gt1, gt2, matches, num_matches = create_ground_truth_correspondences(
            kpts1=outputs['kpt_sampling_grid'][0],
            kpts2=outputs['kpt_sampling_grid'][1],
            deformation=batch_deformation_grid)
        loss_dict = custom_loss(landmark_logits1=outputs['kpt_logits'][0],
                                landmark_logits2=outputs['kpt_logits'][1],
                                desc_pairs_score=outputs['desc_score'],
                                desc_pairs_norm=outputs['desc_norm'],
                                gt1=gt1,
                                gt2=gt2,
                                match_target=matches,
                                k=keypoints,
                                device=device)
        cumu_loss += loss_dict['loss'].item()
        scaler.scale(loss_dict['loss']).backward()
        scaler.step(optimizer)
        scaler.update()

        wandb.log({"batch loss": loss_dict['loss'].item()})

    return cumu_loss / len(loader)

def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        loader = build_dataset(config.batch_size,config.patch_size)
        network = build_network(config.keypoints,config.neighbourhood)
        optimizer = build_optimizer(network, config.optimizer, config.learning_rate,config.L2reg)

        for epoch in range(config.epochs):
            avg_loss = train_epoch(network, loader, optimizer,keypoints=config.keypoints)
            wandb.log({"loss": avg_loss, "epoch": epoch})

def main_hyper():
    sweep_config = {
    'method': 'random'
    }
    metric = {
        'name': 'loss',
        'goal': 'minimize'
        }

    sweep_config['metric'] = metric

    parameters_dict = {
        'optimizer': {
            'values': ['adam']
            },
        'epochs': {
            'values': [20]
            },
        'neighbourhood':{
            'values': [4, 8, 10, 12]
            },
        'keypoints': {
            'values': [128, 256, 512]
        }
        }
    parameters_dict.update({
        'learning_rate': {
            'values': [1e-5, 1e-4, 1e-3]
          },
        'L2reg': {
            'values': [0, 1e-5, 1e-4, 1e-3]
        },
        'batch_size': {
            'values': [2,4,6]
          },
        'patch_size': {
            'values': [(96,96,48)]
        }
        })

    sweep_config['parameters'] = parameters_dict


    import pprint
    pprint.pprint(sweep_config)

    sweep_id = wandb.sweep(sweep_config, project="LandmarkDetect")
    wandb.agent(sweep_id, train, count=20)

if __name__ == '__main__':
    main_hyper()