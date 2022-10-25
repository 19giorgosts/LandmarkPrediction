"""

Script to train landmark correspondence model

See:
    Paper: http://arxiv.org/abs/2001.07434
    Code: https://github.com/monikagrewal/End2EndLandmarks

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""

import os
import sys

sys.path.append(os.path.join(os.path.expanduser('E:/'), 'LandmarkBasedRegistration', 'src', 'util_scripts/'))
sys.path.append(os.path.join(os.path.expanduser('E:/'), 'LandmarkBasedRegistration', 'src', 'arch/'))
# print(sys.path)
from monai.utils import first, set_determinism
from monai.metrics import DiceMetric
from monai.transforms import ShiftIntensity
import torch
import torch.nn as nn
from argparse import ArgumentParser
import shutil
from util_scripts.utils import *
from torch.utils.tensorboard import SummaryWriter
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from util_scripts.pytorchtools import EarlyStopping
from arch.model import Net
from arch.loss import *
from util_scripts.deformations import *
from util_scripts.datapipeline import *
from tqdm import tqdm
import joblib
import torchvision


# Required for CacheDataset
# import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def train(args):
    # Intialize torch GPU
    if args.gpu_id >= 0:
        device = torch.device('cuda:{}'.format(args.gpu_id))
    else:
        device = torch.device('cpu')



    # checkpoint_dir = args.checkpoint_dir

    # log_dir = os.path.join(args.checkpoint_dir, 'logs')
    # log_dir=args.checkpoint_dir

    # if os.path.exists(log_dir) is True:
    #     shutil.rmtree(log_dir)

    # os.makedirs(log_dir)
    # if args.custom:
    #     # print("custom run")
    #     writer = SummaryWriter()
    #     checkpoint_dir = args.checkpoint_dir
    #     if os.path.exists(checkpoint_dir) is False:
    #         os.makedirs(checkpoint_dir)
    # else:
    #     # writer = SummaryWriter("../runs/run_bz{}_pz{}_{}_{}_filt{}_thresh{}".format(args.batch_size, str(args.patch_size[0]), str(args.patch_size[1]), str(args.patch_size[2]), args.filter_scaling,args.thresh))
    #     # checkpoint_dir = args.checkpoint_dir + "run_bz{}_pz{}_{}_{}_filt{}_thresh{}".format(args.batch_size,
    #     #                                                                                     str(args.patch_size[0]),
    #     #                                                                                     str(args.patch_size[1]),
    #     #                                                                                     str(args.patch_size[2]),
    #     #                                                                                     args.filter_scaling,
    #     #                                                                                     args.thresh)
    #     writer = SummaryWriter("../runs/{}".format(args.checkpoint_dir.split('/')[-1]))
    #     checkpoint_dir = args.checkpoint_dir

    summary_path="../runs/{}".format(args.checkpoint_dir.split('/')[-2])
    writer = SummaryWriter("../runs/{}".format(args.checkpoint_dir.split('/')[-2]))
    print(summary_path)
    if os.path.exists(summary_path) is False:
        os.makedirs(summary_path)

    checkpoint_dir = args.checkpoint_dir

    if os.path.exists(checkpoint_dir) is False:
        os.makedirs(checkpoint_dir)

    # Set up data pipeline
    train_patients = joblib.load('train_patients.pkl')
    val_patients = joblib.load('val_patients.pkl')
    print('Number of patients in training set: {}'.format(len(train_patients)))
    print('Number of patients in validation set: {}'.format(len(val_patients)))

    train_dicts = create_data_dicts_dl_reg(train_patients)
    val_dicts = create_data_dicts_dl_reg(val_patients)

    train_loader, _ = create_dataloader_dl_reg(data_dicts=train_dicts,
                                               train=True,
                                               data_aug=True,
                                               shuffle=True,
                                               batch_size=args.batch_size,
                                               num_workers=8,
                                               patch_size=pz)

    val_loader, _ = create_dataloader_dl_reg(data_dicts=val_dicts,
                                             train=True,
                                             data_aug=False,
                                             shuffle=False,
                                             batch_size=1,
                                             num_workers=8,
                                             patch_size=pz)

    model = Net(K=args.K, W=args.W, width=args.filter_scaling, device=device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 1e-4,
                                 weight_decay=1e-4)

    early_stopper = EarlyStopping(patience=args.patience,
                                  path=checkpoint_dir,
                                  delta=1e-5)

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    model.to(device)
    n_iter = 0
    n_iter_val = 0

    print('Start training')
    for epoch in range(100):

        model.train()
        nbatches = len(train_loader)
        pbar = tqdm(enumerate(train_loader), desc="training", total=nbatches, unit="batches")

        for batch_idx, batch_data in pbar:
            images, images_hat, prostate_mask = (batch_data['image'], batch_data['images_hat'], batch_data['prostate_mask'])
            # batch_deformation_grid = get_transform() ## create simple translation transform

            batch_deformation_grid = create_batch_deformation_grid(shape=images.shape,device=images.device,dummy=args.dummy)

            # images_hat = F.grid_sample(input=images,
            #                            grid=batch_deformation_grid,
            #                            align_corners=True,
            #                            mode="bilinear")

            assert (images.shape == images_hat.shape)
            # print(images.shape)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.fp16):
                outputs = model(images.to(device),images_hat.to(device),training=True)
                gt1, gt2, matches = get_labels(pts1=outputs['kpt_sampling_grid'][0],pts2=outputs['kpt_sampling_grid'][1],deformation=batch_deformation_grid)

                loss_dict = custom_loss(landmark_logits1=outputs['kpt_logits'][0],
                                        landmark_logits2=outputs['kpt_logits'][1],
                                        desc_pairs_score=outputs['desc_score'],
                                        desc_pairs_norm=outputs['desc_norm'],
                                        gt1=gt1,
                                        gt2=gt2,
                                        match_target=matches,
                                        k=args.K,
                                        device=device)

            # Backprop
            scaler.scale(loss_dict['loss']).backward()
            scaler.step(optimizer)
            scaler.update()

            ## rotate grids for tensorboard
            images_tb=torch.transpose(images, 2, 4)
            images_hat_tb = torch.transpose(images_hat, 2, 4)
            prostate_mask_tb = torch.transpose(prostate_mask, 2, 4)
            images_tb = torch.flip(images_tb, [3, 4])
            images_hat_tb = torch.flip(images_hat_tb, [3, 4])
            prostate_mask_tb = torch.flip(prostate_mask_tb, [3, 4])

            ## visualize patches
            img_grid0 = torchvision.utils.make_grid(images_tb[0, 0, args.patch_size[2] // 2, :, :])
            img_grid1 = torchvision.utils.make_grid(prostate_mask_tb[0, 0, args.patch_size[2] // 2, :, :])
            img_grid2 = torchvision.utils.make_grid(images_hat_tb[0, 0, args.patch_size[2] // 2, :, :])
            writer.add_image('Training/img1_orig', img_grid0, n_iter)
            writer.add_image('Training/prostate_mask', img_grid1, n_iter)
            writer.add_image('Training/img2_deform', img_grid2, n_iter)

            # save patches in debug mode
            if args.debug:
                sitk.WriteImage(sitk.GetImageFromArray(images_tb[0, 0, :, :, :].detach().numpy()),
                                'debug/output/img{}_{}.gipl'.format(epoch, batch_idx))
                sitk.WriteImage(sitk.GetImageFromArray(images_hat_tb[0, 0, :, :, :].detach().numpy()),
                                'debug/output/deform_{}_{}.gipl'.format(epoch, batch_idx))
                # sitk.WriteImage(sitk.GetImageFromArray(prostate_mask_tb[0, 0, :, :, :].detach().numpy()),
                #                 'debug/output/prostate_mask_{}_{}.gipl'.format(epoch, batch_idx))

            pbar.set_postfix({'Training loss': loss_dict['loss'].item()})
            writer.add_scalar('train/loss', loss_dict['loss'].item(), n_iter)
            writer.add_scalar('train/landmark_1_loss', loss_dict['landmark_1_loss'].item(), n_iter)
            writer.add_scalar('train/landmark_2_loss', loss_dict['landmark_2_loss'].item(), n_iter)
            writer.add_scalar('train/desc_loss_ce', loss_dict['desc_loss_ce'].item(), n_iter)
            writer.add_scalar('train/desc_loss_hinge', loss_dict['desc_loss_hinge'].item(), n_iter)
            # writer.add_scalar('train/desc_loss_hinge_pos', loss_dict['desc_loss_hinge_pos'].item(), n_iter)
            # writer.add_scalar('train/desc_loss_hinge_neg', loss_dict['desc_loss_hinge_neg'].item(), n_iter)
            # writer.add_scalar('train/matches', num_matches, n_iter)
            # writer.add_scalar('train/K', args.K, n_iter)
            n_iter += 1


        print('EPOCH {} done'.format(epoch))

        with torch.no_grad():
            print('Start validation')
            model.eval()
            val_loss = []
            nbatches = len(val_loader)
            pbar_val = tqdm(enumerate(val_loader), desc="validation", total=nbatches, unit="batches")
            for batch_val_idx, val_data in pbar_val:
                images, images_hat, prostate_mask = (val_data['image'], val_data['images_hat'], val_data['prostate_mask'])

                batch_deformation_grid = create_batch_deformation_grid(shape=images.shape,
                                                                       device=images.device,
                                                                       dummy=args.dummy)
                # Folding may have occured
                if batch_deformation_grid is None:
                    continue

                # images_hat = F.grid_sample(input=images,
                #                            grid=batch_deformation_grid,
                #                            align_corners=True,
                #                            mode="bilinear")

                assert (images.shape == images_hat.shape)
                # print(images.shape,images_hat.shape)
                outputs = model(images.to(device),images_hat.to(device),training=True)
                gt1, gt2, matches = get_labels(pts1=outputs['kpt_sampling_grid'][0],pts2=outputs['kpt_sampling_grid'][1],deformation=batch_deformation_grid)
                loss_dict = custom_loss(landmark_logits1=outputs['kpt_logits'][0],
                                        landmark_logits2=outputs['kpt_logits'][1],
                                        desc_pairs_score=outputs['desc_score'],
                                        desc_pairs_norm=outputs['desc_norm'],
                                        gt1=gt1,
                                        gt2=gt2,
                                        match_target=matches,
                                        k=args.K,
                                        device=device)

                # visualize
                landmarks1, landmarks2, true_matches = model.predict(images.to(device), images_hat.to(device))
                # print(images.shape,output1.shape)
                for i in range(images.shape[0]):
                    im1 = images[i, 0, :, :, :].to("cpu").numpy()
                    im2 = images_hat[i, 0, :, :, :].to("cpu").numpy()
                    landmarks1 = landmarks1[i]
                    landmarks2 = landmarks2[i]
                    mask = true_matches[i]
                    found_matches = visualize_keypoints(im1.copy(), im2.copy(), landmarks1, landmarks2, mask, out_dir="../results/{}/".format(args.checkpoint_dir.split('/')[-2]), base_name="iter_{}".format(n_iter_val))

                ## rotate grids for tensorboard
                images_tb = torch.transpose(images, 2, 4)
                images_hat_tb = torch.transpose(images_hat, 2, 4)
                prostate_mask_tb = torch.transpose(prostate_mask, 2, 4)
                images_tb = torch.flip(images_tb, [3, 4])
                images_hat_tb = torch.flip(images_hat_tb, [3, 4])
                prostate_mask_tb = torch.flip(prostate_mask_tb, [3, 4])
                img_grid0 = torchvision.utils.make_grid(images_tb[0, 0, args.patch_size[2] // 2, :, :])
                img_grid1 = torchvision.utils.make_grid(prostate_mask_tb[0, 0, args.patch_size[2] // 2, :, :])
                img_grid2 = torchvision.utils.make_grid(images_hat_tb[0, 0, args.patch_size[2] // 2, :, :])
                writer.add_image('val/img1_orig', img_grid0, n_iter_val)
                writer.add_image('val/prostate_mask', img_grid1, n_iter_val)
                writer.add_image('val/img2_deform', img_grid2, n_iter_val)
                writer.add_scalar('val/loss', loss_dict['loss'].item(), n_iter_val)
                writer.add_scalar('val/landmark_1_loss', loss_dict['landmark_1_loss'].item(), n_iter_val)
                writer.add_scalar('val/landmark_2_loss', loss_dict['landmark_2_loss'].item(), n_iter_val)
                writer.add_scalar('val/desc_loss_ce', loss_dict['desc_loss_ce'].item(), n_iter_val)
                writer.add_scalar('val/desc_loss_hinge', loss_dict['desc_loss_hinge'].item(), n_iter_val)
                # writer.add_scalar('val/matches', num_matches, n_iter_val)
                writer.add_scalar('val/found_matches', found_matches, n_iter_val)
                val_loss.append(loss_dict['loss'].item())
                pbar_val.set_postfix({'Validation loss': loss_dict['loss'].item()})
                n_iter_val += 1


            mean_val_loss = np.mean(np.array(val_loss))

            early_stopper(val_loss=mean_val_loss,
                          model=model)

            # early_stop_condition, best_epoch = early_stopper(val_loss=mean_val_loss,
            #                                                  curr_epoch=epoch,
            #                                                  model=model,
            #                                                  optimizer=optimizer,
            #                                                  scheduler=None,
            #                                                  scaler=scaler,
            #                                                  n_iter=n_iter,
            #                                                  n_iter_val=n_iter_val)

            if early_stopper.early_stop is True:
                # print('Best epoch = {}, stopping training'.format(best_epoch))
                print("Early stopping")
                break


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--K', type=int, default=512)
    parser.add_argument('--W', type=int, default=8)
    parser.add_argument('--thresh', type=int, default=5)
    parser.add_argument('--filter_scaling', type=float, default=1)
    parser.add_argument('--patch_size', nargs='+', type=int, default=(96,96,48))
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--dummy', action='store_true')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--custom', action='store_true', default=False)

    args = parser.parse_args()
    pz=list(args.patch_size)
    # print(pz)
    # print("Using patch size: ", pz)
    train(args)
