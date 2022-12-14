"""

Script with code related to image transformations

A image is transformed in two steps:
    1. Creating a deformation grid (pixel map from original to transformed)
    2. Interpolation (using the inverse map)

Since the deformations will be a part of the loss function, they need to be backprop friendly
so all functions will use torch primitives.

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import numpy as np
import torch
import torch.nn.functional as F
import gryds
import joblib
import os
from datapipeline import *
import SimpleITK as sitk
import shutil

def create_affine_transform(ndim=3,
                            center=[0, 0, 0],
                            angles=[0, 0, 0],
                            scaling=None,
                            shear_matrix=None,
                            translation=None):

    aff_transform = gryds.AffineTransformation(ndim=ndim,
                                               center=center,
                                               angles=angles,
                                               scaling=scaling,
                                               shear_matrix=shear_matrix,
                                               translation=translation)
    return aff_transform

def create_translation_transform(translation):

    trans_transform=gryds.TranslationTransformation(translation)

    return trans_transform

def create_bspline_transform(coarse=True,
                             shape=None,
                             displacements=(5, 3, 3)):
    """

    Displacements are in terms of voxels
    dispacement[k] = m => Along the kth axis, displacements will be in the range (-m , m)

    """

    x, y, z = shape

    if coarse is True:
        random_grid = np.random.rand(3, 4, 4, 4).astype(np.float32) # Make a random 3D 3 x 3 x 3 grid
        random_grid = random_grid*2 - 1 # Shift range to [-1, 1]
        random_grid[0, ...] = random_grid[0, ...]*(2/x)*displacements[0]
        random_grid[1, ...] = random_grid[1, ...]*(2/y)*displacements[1]
        random_grid[2, ...] = random_grid[2, ...]*(2/z)*displacements[2]
    else: # Fine with a larger control grid
        random_grid = np.random.rand(3, 8, 8, 8).astype(np.float32) # Make a random 3D 3 x 3 x 3 grid
        random_grid = random_grid*2 - 1 # Shift range to [-1, 1]
        random_grid[0, ...] = random_grid[0, ...]*(2/x)*displacements[0]
        random_grid[1, ...] = random_grid[1, ...]*(2/y)*displacements[1]
        random_grid[2, ...] = random_grid[2, ...]*(2/z)*displacements[2]

    bspline_transform = gryds.BSplineTransformationCuda(random_grid)

    return bspline_transform

def create_deformation_grid(grid=None,
                            shape=None,
                            transforms=[]):

    if grid is None:
        ndim = len(shape)
        assert(isinstance(transforms, list))

        if ndim == 3:
            grid = np.array(np.meshgrid(np.linspace(-1, 1, shape[0]),
                                        np.linspace(-1, 1, shape[1]),
                                        np.linspace(-1, 1, shape[2]),
                                        indexing="ij"),
                            dtype=np.float32)

            center = [0, 0, 0]
        elif ndim == 2:
            grid = np.array(np.meshgrid(np.linspace(-1, 1, shape[0]),
                                        np.linspace(-1, 1, shape[1]),
                                        indexing="ij"),
                            dtype=np.float32)
            center = [0, 0]


    image_grid = gryds.Grid(grid=grid)

    if len(transforms) == 0: # DEBUG
        return image_grid.grid

    deformed_grid = image_grid.transform(*transforms)

    jac_det = deformed_grid.jacobian_det(*transforms)

    # Check for folding
    if np.amin(jac_det) < 0:
        print('Folding has occured!. Skip this batch')
        return None

    return deformed_grid.grid

import random


def create_batch_deformation_grid(shape,
                                  device='cpu',
                                  dummy=False,
                                  coarse_displacements=(1, 1, 1),
                                  fine_displacements=(0.5, 0.5, 0.5)):

    b, c, i, j, k = shape

    # See: https://discuss.pytorch.org/t/surprising-convention-for-grid-sample-coordinates/79997/6
    # grid[:, h, w, d, 0] = new_x (d)
    # grid[:, h, w, d, 1] = new_y (w)
    # grid[:, h, w, d, 2] = new_z (h)
    batch_deformation_grid = np.zeros((b, i, j, k, 3),
                                      dtype=np.float32)

    # Loop over batch and generated a unique deformation grid for each image in the batch
    for batch_idx in range(b):

        if dummy is False:

            # rand_rotation = random.randrange(0, 10)
            rot_angle=5
            rotation_transform = create_affine_transform(ndim=3,
                                                       angles=[0,0,(np.pi/180)*rot_angle],
                                                       center=[0.5,0.5,0.5],
                                                       )
            # elastic_transform_coarse = create_bspline_transform(coarse=True,
            #                                                     shape=[k, j, i],
            #                                                     displacements=coarse_displacements)
            #
            # elastic_transform_fine = create_bspline_transform(coarse=False,
            #                                                   shape=[k, j, i],
            #                                                   displacements=fine_displacements)
            #
            # transforms = [elastic_transform_coarse, elastic_transform_fine]

            transforms=[rotation_transform]

            # rand_rotation = random.randrange(0, 10)
            # rotation_transform = create_affine_transform(ndim=3,
            #                                            angles=[(np.pi/180)*rand_rotation,(np.pi/180)*rand_rotation,0],
            #                                            center=[0.5,0.5,0.5],
            #                                            )

            ## shift images in axial plane

            # rand_displacement = random.randrange(-10, 10)
            # translation_transform = create_translation_transform(translation=[0,0,0.05*rand_displacement])

            # translation_transform = create_translation_transform(translation=[0.1, 0, 0])
            # transforms=[translation_transform]

        else: # No transforms
            # rand_displacement = random.randrange(-5, 5)
            # translation_transform = create_translation_transform(translation=[0.05 * rand_displacement, 0, 0])
            translation_transform = create_translation_transform(translation=[0.1, 0, 0])
            transforms = [translation_transform]

        # Create deformation grid by composing transforms
        deformed_grid = create_deformation_grid(shape=[k, j, i],
                                                transforms=transforms)

        if deformed_grid is None:
            return None

        # Rearrange axes to make the deformation grid torch-friendly
        ndim, k, j, i = deformed_grid.shape
        deformed_grid = np.reshape(deformed_grid, (ndim, -1)).T
        deformed_grid = np.reshape(deformed_grid, (k, j, i, ndim))

        # Re-order axes to HWD for grid_sample
        deformed_grid = np.transpose(deformed_grid, (2, 1, 0, 3))

        batch_deformation_grid[batch_idx, ...] = deformed_grid

    #Deform the whole batch by stacking all deformation grids along the batch axis (dim=0)
    batch_deformation_grid = torch.Tensor(batch_deformation_grid).to(device)
    # print("image deformed succesfully")
    return batch_deformation_grid



# Test deformations
if __name__ == '__main__':

    # Load all patient paths
    train_patients = joblib.load('../train_patients.pkl')
    train_dict = create_data_dicts_lesion_matching([train_patients[0]])

    data_loader, transforms = create_dataloader_lesion_matching(data_dicts=train_dict,
                                                                train=False,
                                                                batch_size=1,
                                                                data_aug=False)

    post_transforms = Compose([EnsureTyped(keys=['d_image']),
                               Invertd(keys=['d_image',
                                             'liver_mask',
                                             'image',
                                             'vessel_mask'],
                                       transform=transforms,
                                       orig_keys='image',
                                       meta_keys=['d_image_meta_dict',
                                                  'liver_mask_meta_dict',
                                                  'image_meta_dict',
                                                  'vessel_mask_meta_dict'],
                                       nearest_interp=False,
                                       to_tensor=True)])

    for b_id, batch_data in enumerate(data_loader):
        images = batch_data['image']
        batch_deformation_grid = create_batch_deformation_grid(shape=images.shape,
                                                               dummy=True)
        deformed_images = F.grid_sample(input=images,
                                        grid=batch_deformation_grid,
                                        align_corners=True,
                                        mode="nearest")

        sub_image = images - deformed_images

        print(torch.max(sub_image))
        print(torch.min(sub_image))

        assert(torch.allclose(images, deformed_images, atol=1e-4))

        save_dir = 'images_b_{}'.format(b_id)

        if os.path.exists(save_dir) is True:
            shutil.rmtree(save_dir)

        batch_data['d_image'] = deformed_images


        # Save as ITK images with meta-data
        batch_data = [post_transforms(i) for i in decollate_batch(batch_data)]

        save_op_image = SaveImaged(keys='image',
                                   output_postfix='image',
                                   output_dir=save_dir,
                                   separate_folder=False)

        save_op_d_image = SaveImaged(keys='d_image',
                                   output_postfix='d_image',
                                   output_dir=save_dir,
                                   separate_folder=False)

        for batch_dict in batch_data:
            save_op_image(batch_dict)
            save_op_d_image(batch_dict)

