"""
Miscelleneous utility functions

"""
import torch
import imageio
import pickle
import numpy as np
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import label
from skimage.transform import resize
from math import sqrt
import cv2
import pandas as pd
import SimpleITK as sitk


def create_affine_matrix(rotation=0, scale=1, shear=0, translation=0 ,center=np.array([0, 0, 0])):
    """
    Input: rotation angles in degrees
    """
    theta = rotation * np.pi/180
    affine_matrix = np.array([[scale * np.cos(theta), -np.sin(theta + shear), translation],
                [np.sin(theta + shear), scale * np.cos(theta), translation],
                [0, 0, 1]])

    center = center.reshape(-1, 1)
    center_homogenous = np.array([center[0], center[1], center[2]]).reshape(-1, 1)
    center_rotated = np.dot(affine_matrix, center_homogenous)
    # print(center.flatten().shape, center_rotated.flatten()[:3].shape)
    # print(affine_matrix[:3, 2])
    affine_matrix[:3, 2] = center.flatten() - center_rotated.flatten()[:3]
    return affine_matrix.T


def generate_random_2dgaussian(h, w, sigma_h=None, sigma_w=None):
    if sigma_h is None:
        sigma_h = h // 8

    if sigma_w is None:
        sigma_w = w // 8

    H, W = np.meshgrid(np.linspace(0, h, h), np.linspace(0, w, w), indexing="ij")

    center_h, center_w = torch.randint(h // 10, h - h // 10, (1, 1)).item(), torch.randint(w // 10, w - w // 10,
                                                                                           (1, 1)).item()
    sigma_h, sigma_w = torch.randint(sigma_h // 2, sigma_h, (1, 1)).item(), torch.randint(sigma_w // 2, sigma_w,
                                                                                          (1, 1)).item()
    mag_h, mag_w = torch.randint(-4, 4, (1, 1)).item() / 20., torch.randint(-4, 4, (1, 1)).item() / 20.

    if mag_h == 0.:
        mag_h = 0.1
    if mag_w == 0.:
        mag_w = 0.1

    g_h = mag_h * np.exp(-((H - center_h) ** 2 / (2.0 * sigma_h ** 2)))
    g_w = mag_w * np.exp(-((W - center_w) ** 2 / (2.0 * sigma_w ** 2)))

    return g_h.reshape(-1), g_w.reshape(-1)

def generate_deformation_grid(image1):
    """
    Generates a random deformation field, applies it to the input image and returns deformed image and deformation field.

    Inputs:
    image1 = Channels * Height * Width * Depth
    Outputs:
    deformation = Height * Width * * Depth * 3
    """
    print(image1.shape)
    shape = (image1.shape[0], image1.shape[1], image1.shape[2])
    z,y,x = np.meshgrid(np.ones(shape[2]), np.ones(shape[1]), np.ones(shape[0]), indexing="xy")
    indices = np.array([np.reshape(z, -1), np.reshape(y, -1), np.reshape(x, -1)]).T  # shape N, 3
    print(indices.min())
    choices = ["translation"]
    idx = torch.randint(len(choices), (1, 1)).item()
    random_choice = choices[idx]

    if random_choice == "rotation":
        param = (-45, 45)
        angle = torch.randint(param[0], param[1], (1, 1)).item()
        M = create_affine_matrix(rotation=angle)
        indices = np.dot(indices, M)

    elif random_choice == "translation":
        param = (-10, 10)
        range = torch.randint(param[0], param[1], (1, 1)).item()
        M = create_affine_matrix()
        indices = np.dot(indices, M)
        print("dot",indices.min())

    elif random_choice == "scale":
        param = (0.8, 1.2)
        scale = torch.randint(int(param[0] * 100), int(param[1] * 100), (1, 1)).item() / 100.
        M = create_affine_matrix(scale=scale)
        indices = np.dot(indices, M)

    elif random_choice == "shear":
        param = (-20, 20)
        shear = torch.randint(param[0], param[1], (1, 1)).item() * (np.pi / 180.)
        M = create_affine_matrix(shear=shear)
        indices = np.dot(indices, M)

    elif random_choice == "elastic":
        dx, dy = generate_random_2dgaussian(shape[0], shape[1])

        indices[:, 0] += dx
        indices[:, 1] += dy

    # normalized grid for pytorch
    indices = indices[:, :3].reshape(shape[2], shape[1], shape[0], 3)
    # print(indices.shape)
    indices = indices.transpose(0, 1, 2, 3)

    return indices

def resample(image, transform):
    # Output image Origin, Spacing, Size, Direction are taken from the reference
    # image in this call to Resample
    reference_image = image
    interpolator = sitk.sitkNearestNeighbor
    default_value = 0
    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)

def get_transform(dim=3):
    new_transform = sitk.AffineTransform(dim)
    new_transform.SetTranslation((10, 0, 0))
    return new_transform

def affine_translate(grid, dim, x_translation=0, y_translation=0, z_translation=0):
    new_transform = sitk.AffineTransform(dim)
    new_transform.SetTranslation((x_translation, y_translation, z_translation))
    resampled = resample(grid, new_transform)
    return resampled,new_transform

def convert_points_to_torch(pts, X, Y, Z, device="cuda:1"):
    """
    Inputs:-
    pts: k, 2 (Z, Y, X)
    """

    samp_pts = torch.from_numpy(pts.astype(np.float32))
    samp_pts[:, 0] = (samp_pts[:, 0] * 2. / (Z-1)) - 1.
    samp_pts[:, 1] = (samp_pts[:, 1] * 2. / (Y-1)) - 1.
    samp_pts[:, 2] = (samp_pts[:, 2] * 2. / (X-1)) - 1.

    samp_pts = samp_pts.view(1, 1, 1, -1, 3)
    samp_pts = samp_pts.float().to(device)
    return samp_pts

def scale_output(im): ## rescale values to [0,1] for better visual representation
    # Normalized Data
    normalized = (im - im.min()) / (im.max() - im.min())
    return normalized

def zoom(img, zoom_factor=2):
    return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)

def visualize_keypoints(images1, images2, output1, output2, mask, out_dir="./", base_name="im", only_true=False):

    if not only_true:
        pref='val'
    else:
        pref='tr'

    if os.path.exists(out_dir) is False:
        os.makedirs(out_dir)

    # print("\n Saving visualization: ....")
    # print(images1.shape) # (96,96,3), 3rd channel for color
    # pd.DataFrame(mask).to_csv("{}/{}_matches.csv".format(out_dir, base_name), index=None)

    color = [0,0,1]
    # print(output1.shape) # (512,3)
    points = []
    points2 = []
    nonzero_slices=[] ## FIXME based only on im1!
    nonzero_slices2=[]

    ## just saving the points
    for k1, l1, in enumerate(output1):
        points.append(l1)
        z1, _, _ = l1
        if z1 not in nonzero_slices:
            nonzero_slices.append(z1)

    for k2, l2, in enumerate(output2):
        points2.append(l2)
        z2, _, _ = l2
        if z2 not in nonzero_slices2:
            nonzero_slices2.append(z2)

    ## save all points in .txt file for inspection
    ## the coordinates are within the patches! (of course), need sliding window inference otherwise
    with open('{}/{}_{}_im1.txt'.format(out_dir,pref,base_name), 'w') as f:
        for line in points:
            f.write(f"{line}\n")
    with open('{}/{}_{}_im2.txt'.format(out_dir,pref,base_name), 'w') as f:
        for line in points2:
            f.write(f"{line}\n")
    with open('{}/nonzero_slices_{}_im1.txt'.format(out_dir,base_name), 'w') as f:
        for line in sorted(nonzero_slices):
            f.write(f"{line}\n")
    with open('{}/nonzero_slices_{}_im2.txt'.format(out_dir,base_name), 'w') as f:
        for line in sorted(nonzero_slices2):
            f.write(f"{line}\n")

    actual_matches=0
    matches=[]

    # ## visualization starts here
    for slice in nonzero_slices:
        found = False
        # plotting on the correct MR slice
        # print(images1.shape)
        im1=images1[:, :, slice]
        im2=images2[:, :, slice]
        # print(im1.shape)
        img1 = cv2.cvtColor(im1, cv2.COLOR_GRAY2RGB)
        img2 = cv2.cvtColor(im2, cv2.COLOR_GRAY2RGB)
        img = np.concatenate([img1, img2], axis=1)
        for k1, l1, in enumerate(output1):
            z1, y1, x1 = l1 # l1 has (x,y,z) triplet, that's why one more value to unpack!
            if z1==slice:
                if not only_true:
                    cv2.circle(img, (x1, y1), 1, color, -1)
                # print("writing im1 {},{} on slice {}".format(x1,y1,z1))
                for k2, l2, in enumerate(output2):
                    z2, y2, x2 = l2
                    if (z2 == z1): ## if on the same slice (z)
                        if not only_true:
                            cv2.circle(img, (x2+img1.shape[1], y2), 1, color, -1)
                        # print("writing im2 {},{} on slice {}".format(x2, y2, z2))
                        if mask[k1, k2] == 1: ## if real matches
                            found=True
                            if only_true:
                                cv2.circle(img, (x1, y1), 1, color, -1)
                                cv2.circle(img, (x2 + img1.shape[1], y2), 1, color, -1)
                            cv2.line(img, (x1, y1), (x2 + img1.shape[1], y2), color=(0, 1, 0), thickness=1)
                            actual_matches+=1
                            matches.append(((x1,y1),(x2,y2),slice))

        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = zoom(img,4) # zoom into image for better visualization
        img=scale_output(img)
        # print(img.max(),img.min())
        if only_true and found:
            cv2.imwrite(os.path.join(out_dir, "{}_slice_{}.jpg".format(base_name, slice)), (img * 255).astype(np.uint8))
        else:
            cv2.imwrite(os.path.join(out_dir, "{}_slice_{}.jpg".format(base_name, slice)), (img * 255).astype(np.uint8))

        with open('{}/matches_{}.txt'.format(out_dir, base_name), 'w') as f:
            for line in sorted(matches):
                f.write(f"{line}\n")

    return actual_matches