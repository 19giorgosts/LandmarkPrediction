"""

A model to find correspondences (landmarks) between a pair of images.
Contrary to traditional approaches, this model performs detection and description
in parallel (using different layers of the same network)

Reference:
    Paper: http://arxiv.org/abs/2109.02722
    Code: https://github.com/monikagrewal/End2EndLandmarks

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from unet3d import UNet
from torchsummary import summary
from util_scripts.utils import *

def weight_init(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.fill_(0.0)
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class Net(nn.Module):

    def __init__(self,
                 K=512,
                 W=4,
                 n_channels=1,
                 width=1,
                 device="cuda:0"
                 ):
        super(Net, self).__init__()

        # Main learning block that jointly performs description (intermediate feature maps) and detection (output map)
        self.cnn = UNet(n_channels=n_channels, n_classes=1, trilinear=False, width_multiplier=width)
        # print(summary(self.cnn, (1, 96, 96, 48), device="cpu"))
        self.desc_matching_layer = DescriptorMatcher(in_channels=self.cnn.descriptor_length, out_channels=2)

        self.W = W
        self.K = K
        self.device=device

    def forward(self, x1, x2, training=True):

        # landmark detection and description
        kpts_1, features_1 = self.cnn(x1)
        kpts_2, features_2 = self.cnn(x2)

        # Sample keypoints and corr. descriptors for I1 and I2 respectively
        kpt_sampling_grid_1, kpt_logits_1, descriptors_1 = self.sampling_block(heatmaps=kpts_1,features=features_1,is_training=training)
        kpt_sampling_grid_2, kpt_logits_2, descriptors_2 = self.sampling_block(heatmaps=kpts_2,features=features_2,is_training=training)

        # descriptor matching probabilities and descriptor norms
        desc_pairs_score, desc_pair_norm = self.desc_matching_layer(descriptors_1,descriptors_2)

        output_dict = {}
        output_dict['kpt_logits'] = (kpt_logits_1, kpt_logits_2)
        output_dict['kpt_sampling_grid'] = (kpt_sampling_grid_1, kpt_sampling_grid_2)
        output_dict['desc_score'] = desc_pairs_score
        output_dict['desc_norm'] = desc_pair_norm

        return output_dict

    # Break the 'inference' function into CNN output and sampling+matching
    # so that sliding_window_inference can be performed
    # We need to run the model twice because sliding_window_inference
    # expects all outputs to be of the same size
    # THESE FUNCTIONS ARE USED ONLY DURING INFERENCE!!!!
    def get_patch_keypoint_scores(self, x):

        x1 = x[:, 0, ...].unsqueeze(dim=1)
        x2 = x[:, 1, ...].unsqueeze(dim=1)

        # 1. Landmark (candidate) detections (logits)
        kpts_1, _ = self.cnn(x1)
        kpts_2, _ = self.cnn(x2)

        return kpts_1, kpts_2

    def get_patch_feature_descriptors(self, x):

        x1 = x[:, 0, ...].unsqueeze(dim=1)
        x2 = x[:, 1, ...].unsqueeze(dim=1)

        # 1. Landmark (candidate) detections
        _, features_1 = self.cnn(x1)
        _, features_2 = self.cnn(x2)

        # "features" is a tuple of feature maps from different U-Net resolutions
        # We return separate tensors so 'sliding_window_inference' works
        return features_1[0], features_1[1], features_2[0], features_2[1]


    def inference(self, kpts_1, kpts_2, features_1, features_2, conf_thresh=0.5, num_pts=512):

        b, c, i, j, k = kpts_1.shape
        # print(k,j,i)
        # 2.)Sampling grid + descriptors
        kpt_sampling_grid_1, kpt_logits_1, descriptors_1 = self.sampling_block(heatmaps=kpts_1,
                                                                               features=features_1,
                                                                               is_training=False,
                                                                               conf_thresh=conf_thresh)

        kpt_sampling_grid_2, kpt_logits_2, descriptors_2 = self.sampling_block(heatmaps=kpts_2,
                                                                               features=features_2,
                                                                               is_training=False,
                                                                               conf_thresh=conf_thresh)

        # 3. Compute descriptor matching scores
        desc_pairs_score, desc_pair_norm = self.descriptor_matching(descriptors_1, descriptors_2)

        landmarks_1 = self.convert_grid_to_image_coords(kpt_sampling_grid_1, shape=(i,j,k))
        landmarks_2 = self.convert_grid_to_image_coords(kpt_sampling_grid_2, shape=(i,j,k))

        _, k1, k2 = desc_pair_norm.shape
        print(desc_pairs_score.shape)
        # Match probability
        desc_pairs_prob = F.softmax(desc_pairs_score, dim=1)[:, 1].view(b, k1, k2)

        # Two-way matching
        matches = []
        matches_norm = []
        matches_prob = []

        for batch_idx in range(b):

            pairs_prob = desc_pairs_prob[batch_idx]
            pairs_norm = desc_pair_norm[batch_idx]
            # print(pairs_prob)

            # 2-way matching w.r.t pair probabilities
            match_cols = torch.zeros((k1, k2))
            match_cols[torch.argmax(pairs_prob, dim=0), torch.arange(k2)] = 1
            # print(match_cols)
            match_rows = torch.zeros((k1, k2))
            match_rows[torch.arange(k1), torch.argmax(pairs_prob, dim=1)] = 1
            match_prob = match_rows*match_cols

            # 2-way matching w.r.t probabilities & min norm
            match_cols = torch.zeros((k1, k2))
            match_cols[torch.argmin(pairs_norm, dim=0), torch.arange(k2)] = 1
            match_rows = torch.zeros((k1, k2))
            match_rows[torch.arange(k1), torch.argmin(pairs_norm, dim=1)] = 1
            match_norm = match_rows*match_cols

            match = match_prob*match_norm

            matches.append(match)
            matches_norm.append(match_norm)
            matches_prob.append(match_prob)

        matches = torch.stack(matches)
        matches_norm = torch.stack(matches_norm)
        matches_prob = torch.stack(matches_prob)

        outputs = {}
        outputs['landmarks_1'] = landmarks_1
        outputs['landmarks_2'] = landmarks_2
        outputs['matches'] = matches
        outputs['matches_norm'] = matches_norm
        outputs['matches_prob'] = matches_prob
        outputs['kpt_sampling_grid_1'] = kpt_sampling_grid_1
        outputs['kpt_sampling_grid_2'] = kpt_sampling_grid_2

        return outputs

    def predict(self, x1, x2, deformation=None, conf_thresh=0.5, k=None):
        if k is None:
            k = self.K
        scale_factor = self.W
        b, _, X, Y, Z = x1.shape
        # landmark detection and description
        heatmaps1, features1 = self.cnn(x1)
        heatmaps2, features2 = self.cnn(x2)

        # sampling top k landmark locations and descriptors
        pts1, _, desc1 = self.sampling_block(heatmaps1, features1, conf_thresh=conf_thresh, is_training=False)
        pts2, _, desc2 = self.sampling_block(heatmaps2, features2, conf_thresh=conf_thresh, is_training=False)

        # descriptor matching probabilities and descriptor norms
        desc_pairs_score, desc_pairs_norm = self.desc_matching_layer(desc1, desc2)

        # post processing
        landmarks1 = self.convert_points_to_image(pts1, X, Y, Z)
        landmarks2 = self.convert_points_to_image(pts2, X, Y, Z)

        b, k1, _ = landmarks1.shape
        _, k2, _ = landmarks2.shape

        # two-way (bruteforce) matching
        desc_pairs_score = F.softmax(desc_pairs_score, dim=1)[:, 1].view(b, k1, k2)
        desc_pairs_score = desc_pairs_score.detach().to("cpu").numpy()
        desc_pairs_norm = desc_pairs_norm.detach().to("cpu").numpy()
        matches = list()
        for i in range(b):
            pairs_score = desc_pairs_score[i]
            pairs_norm = desc_pairs_norm[i]

            match_cols = np.zeros((k1, k2))
            match_cols[np.argmax(pairs_score, axis=0), np.arange(k2)] = 1
            match_rows = np.zeros((k1, k2))
            match_rows[np.arange(k1), np.argmax(pairs_score, axis=1)] = 1
            match = match_rows * match_cols

            match_cols = np.zeros((k1, k2))
            match_cols[np.argmin(pairs_norm, axis=0), np.arange(k2)] = 1
            match_rows = np.zeros((k1, k2))
            match_rows[np.arange(k1), np.argmin(pairs_norm, axis=1)] = 1
            match = match * match_rows * match_cols

            matches.append(match)

        matches = np.array(matches)

        if deformation is not None:
            deformation = deformation.permute(0, 4, 1, 2, 3)  # b, 2, h, w
            pts1_projected = F.grid_sample(deformation, pts2)  # b, 2, 1, k
            pts1_projected = pts1_projected.permute(0, 2, 3, 1)  # b, 1, k, 2
            landmarks1_projected = self.convert_points_to_image(pts1_projected, X, Y, Z)
            return landmarks1, landmarks2, matches, landmarks1_projected
        else:
            return landmarks1, landmarks2, matches

    # @staticmethod
    # def convert_grid_to_image_coords(pts, shape=(64, 128, 128)):
    #     """
    #     Convert probabilities ([-1, 1] range) to image coords
    #
    #     """
    #     # print("shape", shape)
    #     pts = pts.squeeze(dim=1).squeeze(dim=1) # Shape: [B, K, 3]
    #
    #     # Scale to [0, 1] range
    #     pts = (pts + 1.)/2.
    #
    #     # Scale to image dimensions (DHW ordering)
    #     pts = pts * torch.Tensor([(shape[2]-1, shape[1]-1, shape[0]-1)]).view(1, 1, 3).to(pts.device)
    #
    #     return pts

    # @staticmethod
    # def convert_grid_to_image_coords(pts, shape=(64, 128, 128)):
    #     """
    #     Convert grid points ([-1, 1] range) to image coords
    #     """
    #
    #     pts = pts.squeeze(dim=1).squeeze(dim=1)  # Shape: [B, K, 3]
    #
    #     # Scale to [0, 1] range
    #     pts = (pts + 1.) / 2.
    #
    #     # Scale to image dimensions (DHW ordering)
    #     pts = pts * torch.Tensor([(shape[0] - 1, shape[1] - 1, shape[2] - 1)]).view(1, 1, 3).to(pts.device)
    #
    #     return pts

    @staticmethod
    def convert_points_to_image(samp_pts, X, Y, Z):
        """
        Inputs:  samp_pts: b, 1, 1, k, 3
        Outputs: samp_pts: b, k, 3
        """
        # print(samp_pts.shape)
        b, _, _, K, _ = samp_pts.shape
        # Convert pytorch -> numpy.
        samp_pts = samp_pts.data.cpu().numpy().reshape(b, K, 3)
        samp_pts = (samp_pts + 1.) / 2.
        samp_pts = samp_pts * np.array([float(Z - 1), float(Y - 1), float(X - 1)]).reshape(1, 1, 3)
        return samp_pts.astype(np.int32)


    def sampling_block(self, heatmaps, features, conf_thresh=0.1, is_training=True):
        k = self.K
        W = self.W

        b, _, X, Y, Z = heatmaps.shape
        heatmaps = torch.sigmoid(heatmaps)

        """
        Convert pytorch -> numpy after maxpooling and unpooling
        This is faster way of sampling while ensuring sparsity
        One could alternatively apply non-maximum suppresion (NMS)
        """
        if is_training:
            heatmaps1, indices = F.max_pool3d(heatmaps, (W, W, W), stride=(W, W, W), return_indices=True)
            heatmaps1 = F.max_unpool3d(heatmaps1, indices, (W, W, W))
            heatmaps1 = heatmaps1.to("cpu").detach().numpy().reshape(b, X, Y, Z)
        else:
            heatmaps1 = heatmaps.to("cpu").detach().numpy().reshape(b, X, Y, Z)

        # border mask, optional
        border = 10
        border_mask = np.zeros_like(heatmaps1)
        border_mask[:, border: X - border, border: Y - border, border: Z - border] = 1.
        heatmaps1 = heatmaps1 * border_mask
        all_pts = []
        for heatmap in heatmaps1:
            xs, ys, zs = np.where(heatmap >= conf_thresh)  # get landmark locations above conf_thresh
            if is_training:
                if len(xs) < k:
                    print('Number of point above threshold {} are less than K={}'.format(len(xs), k))
                    xs, ys, zs = np.where(heatmap >= 0.0)
            pts = np.zeros((len(xs), 4))
            pts[:, 0] = zs
            pts[:, 1] = ys
            pts[:, 2] = xs
            pts[:, 3] = heatmap[xs, ys, zs]

            inds = np.argsort(pts[:, 3])
            pts = pts[inds[::-1], :]  # sort by probability scores
            pts = pts[:k, :3]  # take top k

            # Interpolate into descriptor map using 3D point locations.
            samp_pts = convert_points_to_torch(pts, X, Y, Z, device=self.device)
            all_pts.append(samp_pts)

        all_pts = torch.cat(all_pts, dim=0)
        pts_score = F.grid_sample(heatmaps, all_pts)  # b, 1, 1, 1, k
        pts_score = pts_score.squeeze(dim=1).squeeze(dim=1).squeeze(dim=1)
        # pts_score = pts_score.permute(0, 4, 1, 2, 3).view(b, -1)
        desc = [F.grid_sample(desc, all_pts) for desc in features]
        desc = torch.cat(desc, dim=1)
        return all_pts, pts_score, desc


class DescriptorMatcher(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels):

        super(DescriptorMatcher, self).__init__()

        self.fc = nn.Linear(in_channels, out_channels)
        self.apply(weight_init) # weight initialization


    def forward(self, out1, out2):

        b, c, d1, h1, w1 = out1.size()
        _, _, d2, h2, w2 = out2.size()

        # print(out1.shape, out2.shape)

        out1 = out1.view(b, c, d1*h1*w1).permute(0, 2, 1).view(b, d1*h1*w1, 1, c)
        out2 = out2.view(b, c, d2*h2*w2).permute(0, 2, 1).view(b, 1, d2*h2*w2, c)

        # print(out1.shape, out2.shape)
        # Outer product to get all possible pairs
        # Shape: [b, k1, k2, c]
        out = out1*out2
        out = out.contiguous().view(-1, c)

        # Unnormalized logits used for CE loss
        # Alternatively can be single channel with BCE loss
        out = self.fc(out)

        # Compute norms
        desc_l2_norm_1 = torch.norm(out1, p=2, dim=3)
        out1_norm = out1.div(1e-6 + torch.unsqueeze(desc_l2_norm_1, dim=3))

        desc_l2_norm_2 = torch.norm(out2, p=2, dim=3)
        out2_norm = out2.div(1e-6 + torch.unsqueeze(desc_l2_norm_2, dim=3))

        out_norm = torch.norm(out1_norm-out2_norm, p=2, dim=3)

        return out, out_norm



