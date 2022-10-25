"""

Preprocessing pipeline

@author:Georgios Tsekas
@email: gtsekas@umcutrecht.nl

"""

import os
import random
from argparse import ArgumentParser
from image_utils import *
from split_dataset import *

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dummy', action='store_true')
    args = parser.parse_args()
    # extract_prostate_mask(args.data_dir)
    # convert_to_nii(args.data_dir)
    gryds_transform(args)

