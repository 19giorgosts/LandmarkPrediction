"""

Script to evaluate trained models (with visualization)

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
from monai.utils import first, set_determinism
from monai.metrics import DiceMetric
from monai.transforms import ShiftIntensity
from monai.inferers import sliding_window_inference
import torch
import torch.nn as nn
from argparse import ArgumentParser
import shutil
from util_scripts.utils import *
import os, sys
sys.path.append(os.path.join(os.path.expanduser('E:/'), 'LandmarkBasedRegistration', 'src', 'util_scripts/'))
sys.path.append(os.path.join(os.path.expanduser('E:/'), 'LandmarkBasedRegistration', 'src', 'arch/'))
from arch.model import Net
from util_scripts.deformations import *
from util_scripts.datapipeline import *
from arch.loss import create_ground_truth_correspondences
from util_scripts.metrics import get_match_statistics
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# def plot_matches(grid,points):
from util_scripts.visualize import save_matches

def test(args):

    # Intialize torch GPU
    if args.gpu_id >= 0:
        device = torch.device('cuda:{}'.format(args.gpu_id))
    else:
        device = torch.device('cpu')

    checkpoint_dir = args.checkpoint_dir
    output_path = args.result_path + args.checkpoint_dir.split('/')[-2]
    if os.path.exists(output_path) is False:
        os.makedirs(output_path)

    # Set up data pipeline
    if args.mode == 'val':
        patients = joblib.load('val_patients.pkl')
    elif args.mode == 'test':
        patients = joblib.load('test_patients.pkl')
        print(patients)
    elif args.mode == 'train':
        patients = joblib.load('train_patients.pkl')


    if args.synthetic is True:
        data_dicts = create_data_dicts_dl_reg(patients)
        data_loader, _ = create_dataloader_dl_reg(data_dicts=data_dicts,
                                                          train=False,
                                                          batch_size=args.batch_size,
                                                          num_workers=4,
                                                          data_aug=False)
    else: # "Real" data
        data_dicts = create_data_dicts_lesion_matching_inference(patients)
        data_loader, _ = create_dataloader_lesion_matching_inference(data_dicts=data_dicts,
                                                                     batch_size=args.batch_size,
                                                                     num_workers=4)


    # Define the model
    model = Net(W=args.W, width=args.filter_scaling, device=args.gpu_id)

    # Load the model

    # load_dict = load_model(model=model,
    #                        checkpoint_dir=checkpoint_dir,
    #                        training=False)
    #
    # model = load_dict['model']
    model.load_state_dict(torch.load(checkpoint_dir+'checkpoint.pt'))
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            print("Test image ", batch_idx)
            images, prostate_mask = (batch_data['image'], batch_data['prostate_mask'])
            print(images.shape,prostate_mask.shape)
            # batch_deformation_grid = create_batch_deformation_grid(shape=images.shape,
            #                                                        device=images.device,
            #                                                        dummy=args.dummy,
            #                                                        coarse_displacements=(6, 3, 3),
            #                                                        fine_displacements=(2, 2, 2))

            batch_deformation_grid = create_batch_deformation_grid(shape=images.shape,
                                                                   device=images.device,
                                                                   dummy=args.dummy)

            if batch_deformation_grid is None:
                continue


            images_hat = F.grid_sample(input=images,
                                       grid=batch_deformation_grid,
                                       align_corners=True,
                                       mode="bilinear")

            # if args.debug:
            #     sitk.WriteImage(sitk.GetImageFromArray(images_tb[0, 0, :, :, :].detach().numpy()),
            #                     'debug/output/img{}_{}.gipl'.format(epoch, batch_idx))
            #     sitk.WriteImage(sitk.GetImageFromArray(images_hat_tb[0, 0, :, :, :].detach().numpy()),
            #                     'debug/output/deform_{}_{}.gipl'.format(epoch, batch_idx))
            #     sitk.WriteImage(sitk.GetImageFromArray(prostate_mask_tb[0, 0, :, :, :].detach().numpy()),
            #                     'debug/output/prostate_mask_{}_{}.gipl'.format(epoch, batch_idx))

            # Concatenate along channel axis so that sliding_window_inference can
            # be used

            assert(images_hat.shape == images.shape)
            images_cat = torch.cat([images, images_hat], dim=1)

            # Pad so that sliding window inference does not complain
            # about non-integer output shapes
            depth = images_cat.shape[-1]
            pad = 256-depth
            images_cat = F.pad(images_cat, (0, pad), "constant", 0)
            prostate_mask = F.pad(prostate_mask, (0, pad), "constant", 0)

            # save images & deformed images for visual inspection
            save_im=torch.transpose(images_cat, 2, 4)
            save_im=torch.flip(save_im, [3, 4])
            save_im1=sitk.GetImageFromArray(save_im[0, 0, :].detach().numpy())
            save_im1.SetSpacing([2.0,2.0,2.0])
            sitk.WriteImage(save_im1, output_path + '/test_img_{}_1.gipl'.format(batch_idx))
            save_im2=sitk.GetImageFromArray(save_im[0, 1, :].detach().numpy())
            save_im2.SetSpacing([2.0,2.0,2.0])
            sitk.WriteImage(save_im2, output_path + '/test_img_{}_2.gipl'.format(batch_idx))


            # # Keypoint logits
            # kpts_1, kpts_2 = sliding_window_inference(inputs=images_cat.to(device),roi_size=pz,sw_batch_size=1,predictor=model.get_patch_keypoint_scores,overlap=0.5)
            # # Mask using prostate mask
            # kpts_1 = kpts_1*prostate_mask.to(kpts_1.device)
            # kpts_2 = kpts_2*prostate_mask.to(kpts_2.device)
            # # Feature maps
            # features_1_low, features_1_high, features_2_low, features_2_high =sliding_window_inference(inputs=images_cat.to(device),roi_size=pz,sw_batch_size=1,predictor=model.get_patch_feature_descriptors,
            #                                                                  overlap=0.5)
            # features_1 = (features_1_low, features_1_high)
            # features_2 = (features_2_low, features_1_high)
            # # Get (predicted) landmarks and matches on the full image
            # # These landmarks are predicted based on L2-norm between feature descriptors
            # # and predicted matching probability
            # outputs = model.inference(kpts_1=kpts_1,
            #                           kpts_2=kpts_2,
            #                           features_1=features_1,
            #                           features_2=features_2,
            #                           conf_thresh=0.5)
            # Get ground truth matches based on projecting keypoints using the deformation grid
            gt1, gt2, gt_matches, num_gt_matches = create_ground_truth_correspondences(kpts1=outputs['kpt_sampling_grid_1'],
                                                                                       kpts2=outputs['kpt_sampling_grid_2'],
                                                                                       deformation=batch_deformation_grid,
                                                                                       pixel_thresh=5)

            print('Number of ground truth matches (based on projecting keypoints) = {}'.format(num_gt_matches))
            print('Number of matches based on feature descriptor distance '
                  '& matching probability = {}'.format(torch.nonzero(outputs['matches']).shape[0]))

            # Get TP, FP, FN matches
            for batch_id in range(gt_matches.shape[0]):
                batch_gt_matches = gt_matches[batch_id, ...] # Shape: (K, K)
                batch_pred_matches = outputs['matches'][batch_id, ...] # Shape (K, K)
                batch_pred_matches_norm = outputs['matches_norm'][batch_id, ...] # Shape (K, K)
                batch_pred_matches_prob = outputs['matches_prob'][batch_id, ...] # Shape (K, K)

                stats = get_match_statistics(gt=batch_gt_matches.cpu(),
                                             pred=batch_pred_matches.cpu())
                print('Matches :: TP matches = {} FP = {} FN = {}'.format(stats['True Positives'],
                                                                          stats['False Positives'],
                                                                          stats['False Negatives']))

                stats = get_match_statistics(gt=batch_gt_matches.cpu(),
                                             pred=batch_pred_matches_norm.cpu())
                print('Matches w.r.t L2-Norm:: TP matches = {} FP = {} FN = {}'.format(stats['True Positives'],
                                                                                       stats['False Positives'],
                                                                                       stats['False Negatives']))

                stats = get_match_statistics(gt=batch_gt_matches.cpu(),
                                             pred=batch_pred_matches_prob.cpu())
                print('Matches w.r.t match probability:: TP matches = {} FP = {} FN = {}'.format(stats['True Positives'],
                                                                                                 stats['False Positives'],
                                                                                                 stats['False Negatives']))
                print("\n")
                save_matches(outputs,batch_idx,output_path)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--patch_size', nargs='+', type=int, default=(96, 96, 48))
    parser.add_argument('--filter_scaling', type=float, default=1)
    parser.add_argument('--dummy', action='store_true', default=False)
    parser.add_argument('--W', type=int, default=8)
    parser.add_argument('--result_path', type=str, default="E:/LandmarkBasedRegistration/results/")


    args = parser.parse_args()
    pz = list(args.patch_size)

    test(args)
