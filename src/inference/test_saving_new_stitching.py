import time
import os
import torch
import monai
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from options.test_options import TestOptions
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
# from data.dataset_TEST import CreateDataloader_TEST
from data.dataset_CACHE_Whole_body import CreateDataloader  # data.dataset_CACHE
# from data.dataset_CACHE import CreateDataloader
from monai.transforms import SpatialCropd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import imageio
from scipy.ndimage import gaussian_filter
import pandas as pd
import nibabel as nib
import re
from nilearn.image import resample_to_img
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

# ==== Parameters ====
patch_dim = 32
overlap_dim = 16

# ==== Device ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

opt = TestOptions().parse()  # impostare phase == test

opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataloader(opt, shuffle=False, cache=False)  # CreateDataloader_TEST
dataset_size = len(data_loader)
print('#testing images = %d' % dataset_size)

out_path = '/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/ct2pet/generated_data/ENHANCE_PET_curriculumGAN/test_new_stitching'  # 'mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/CTPET_train'  /mimer/NOBACKUP/groups/snic2022-5-277/piacente/test_external_generation_output

model = create_model(opt)

# test
for i, data in enumerate(data_loader):

    A, B = data["A"], data["B"]
    print(A.shape, B.shape, data.keys())

    image_shape = data['A'].shape[2:]

    os.makedirs(out_path, exist_ok=True)
    # match = re.search(r'(PETCT_\w{10})', data["A_paths"][0])
    # match = re.search(r'(Sqc_\w+|Adk_\w+|ADK \w+)', data["A_paths"][0])
    # subject_id = match.group(1) if match else f"unknown_{i+1}"

    # Extract subject ID as the last folder name (e.g. "1597")
    subject_id = os.path.basename(os.path.dirname(os.path.normpath(data["A_paths"][0])))

    model.set_input(data)
    model.test_sliding()
    visuals = model.get_current_visuals()
    fake_B_test = visuals['fake_B'].squeeze()

    image_filename_nifti = f"{subject_id}_generated_image_{i + 1}.nii.gz"
    image_path_nifti = os.path.join(out_path, image_filename_nifti)
    complete_generated_image_nifti = nib.Nifti1Image(fake_B_test, np.eye(4))
    nib.save(complete_generated_image_nifti, image_path_nifti)


