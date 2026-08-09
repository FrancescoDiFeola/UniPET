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

# ==== Parameters ====
patch_dim = 32
overlap_dim = 16

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

def resample_nifti_to_spacing(nifti_path, target_spacing, interpolation_order=1):
    """
    Resample a NIfTI image to the desired voxel spacing.

    Args:
        nifti_path (str): Path to the input NIfTI file.
        target_spacing (tuple): Desired spacing in (x, y, z) order, e.g., (2.0, 2.0, 3.0).
        interpolation_order (int): 0=nearest, 1=linear, 3=cubic.

    Returns:
        nib.Nifti1Image: Resampled image with updated affine.
    """
    # Load image
    img = nib.load(nifti_path)
    data = img.get_fdata()
    original_spacing = img.header.get_zooms()[:3]  # (x, y, z)
    print(original_spacing)
    # Compute zoom factor (z, y, x) due to data shape order
    zoom_factors = tuple(osz / tsz for osz, tsz in zip(original_spacing, target_spacing))

    # Apply zoom
    resampled_data = zoom(data, zoom=zoom_factors, order=interpolation_order)

    # Create new affine
    new_affine = np.copy(img.affine)
    for i in range(3):
        new_affine[i, i] = target_spacing[i]

    # Wrap in NIfTI
    # resampled_img = nib.Nifti1Image(resampled_data, affine=new_affine)

    return resampled_data, new_affine
    
def patch_indices(shape, patch=patch_dim, overlap=overlap_dim):
    indices, step = [], patch - overlap
    for x in range(0, shape[0], step):
        for y in range(0, shape[1], step):
            for z in range(0, shape[2], step):
                cx = min(x + patch // 2, shape[0] - overlap)
                cy = min(y + patch // 2, shape[1] - overlap)
                cz = min(z + patch // 2, shape[2] - overlap)
                indices.append((cx, cy, cz))
    return indices


def save_gif(image_3d, path, duration=0.1):  # Salva una sequenza di immagini 2D come una GIF animata.
    print(f"save_gif: shape di image_3d = {image_3d.shape}")
    depth = image_3d.shape[1]
    try:
        frames = [image_3d[:, i, :] for i in range(depth)]
        imageio.mimsave(path, frames, duration=duration)
    except IndexError as e:
        print(f"IndexError: {e}")
        print(f"Forma dell'immagine: {image_3d.shape}")


def scale_image(image):
    # Riscalamento dell'immagine nell'intervallo 0-1
    scaled_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    # Riscalamento nell'intervallo 0-255
    scaled_image = (scaled_image * 255).astype(np.uint8)
    return scaled_image


def scale_fake_with_original(fake_image, original_image):
    # Riscalamento dell'immagine fake nell'intervallo 0-1 usando min e max dell'immagine originale
    scaled_fake_image = (fake_image - np.min(original_image)) / (np.max(original_image) - np.min(original_image))
    # Riscalamento dell'immagine fake nell'intervallo 0-255
    scaled_fake_image = (scaled_fake_image * 255).astype(np.uint8)
    return scaled_fake_image


def denorm(tensor, a_min=0, a_max=20):
    return tensor * (a_max - a_min) + a_min


def compute_pet_metrics(suv_values):
    suv_values = suv_values.flatten()
    SUV_max = np.max(suv_values)
    SUV_mean = np.mean(suv_values)
    MTV_15 = np.sum(suv_values >= 1.5)
    MTV_25 = np.sum(suv_values >= 2.5)
    TLG_15 = MTV_15 * SUV_mean
    TLG_25 = MTV_25 * SUV_mean
    return {
        'SUV_max': SUV_max,
        'SUV_mean': SUV_mean,
        'MTV_1.5': MTV_15,
        'TLG_1.5': TLG_15,
        'MTV_2.5': MTV_25,
        'TLG_2.5': TLG_25
    }

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

model = create_model(opt)
visualizer = Visualizer(opt)

mae_per_patient = []
psnr_per_patient = []
ssim_per_patient = []
all_results = []
# test
for i, data in enumerate(data_loader):

    A, B = data["A"], data["B"]
    print(A.shape, B.shape, data.keys())

    # img_A, _ = resample_nifti_to_spacing(data["A_paths"][0], (2.03642, 2.03642, 3))
    # img_B, _ = resample_nifti_to_spacing(data["B_paths"][0], (2.03642, 2.03642, 3))
    # data["A"] = img_A[None, None, ...]  # torch.from_numpy(img_A).unsqueeze(0).unsqueeze(0)  # Add [B, C]
    # data["B"] = img_B[None, None, ...]  # torch.from_numpy(img_B).unsqueeze(0).unsqueeze(0)
    # A, B = data["A"], data["B"]
    # print(f"DATA: {data}")

    image_shape = data['A'].shape[2:]
    original_CT = data['A'].as_tensor().permute(0, 2, 3, 4, 1)  # torch.from_numpy(data['A']).permute(0, 2, 3, 4, 1)
    original_CT_3d = original_CT.numpy().squeeze(axis=(0, 4))
    out_path = '/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/ct2pet/generated_data/FDG_CTPET_curriculumGAN/test_set'   # 'mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/CTPET_train'  /mimer/NOBACKUP/groups/snic2022-5-277/piacente/test_external_generation_output
    os.makedirs(out_path, exist_ok=True)
    match = re.search(r'(PETCT_\w{10})', data["A_paths"][0])
    # match = re.search(r'(Sqc_\w+|Adk_\w+|ADK \w+)', data["A_paths"][0])
    subject_id = match.group(1) if match else f"unknown_{i+1}"
    image_filename_CT = f"{subject_id}_original_CT_{i+1}.nii.gz"
    image_path_original_CT_nifti = os.path.join(out_path, image_filename_CT)

    image_filename_original_nifti = f"{subject_id}_original_image_{i + 1}.nii.gz"  # Puoi personalizzare il nome del file come desideri
    image_path_original_nifti = os.path.join(out_path, image_filename_original_nifti)
    image_filename_nifti = f"{subject_id}_generated_image_{i + 1}.nii.gz"  # Puoi personalizzare il nome del file come desideri
    image_path_nifti = os.path.join(out_path, image_filename_nifti)
    
    # Skip processing if all output files already exist
    if (
            os.path.exists(image_path_original_CT_nifti)
            and os.path.exists(image_path_nifti)
            and os.path.exists(image_path_original_nifti)
    ):
        print(f"Skipping {subject_id}: outputs already exist.")
        continue
    original_CT_nifti = nib.Nifti1Image(original_CT_3d, np.eye(4))
    nib.save(original_CT_nifti, image_path_original_CT_nifti)
    shape = A.shape[2:]
    patches = patch_indices(shape)
    _, _, D, H, W = A.shape

    generated = np.zeros((D, H, W), dtype=np.float32)
    overlap = np.zeros((D, H, W), dtype=np.float32)

    for center in patches:
        crop = SpatialCropd(keys=["A", "B"], roi_center=center, roi_size=[patch_dim] * 3)
        patches = crop({
            "A": A.as_tensor().permute(0, 2, 3, 4, 1),  # torch.from_numpy(data['A']).permute(0, 2, 3, 4, 1)
            "B": B.as_tensor().permute(0, 2, 3, 4, 1)
        })
        patches['A'] = patches['A'].squeeze().unsqueeze(0).unsqueeze(0).to(device)
        patches['B'] = patches['B'].squeeze().unsqueeze(0).unsqueeze(0).to(device)

        model.set_input(patches)
        model.test()
        visuals = model.get_current_visuals()

        real_B_test = visuals['real_B']
        fake_B_test = visuals['fake_B'].squeeze() # .cpu().numpy()

        cx, cy, cz = center
        sx = sy = sz = overlap_dim
        generated[cx - sx:cx + sx, cy - sy:cy + sy, cz - sz:cz + sz] += fake_B_test
        overlap[cx - sx:cx + sx, cy - sy:cy + sy, cz - sz:cz + sz] += 1

    generated /= np.maximum(overlap, 1)
    fake_image = generated
    real = B.squeeze().numpy()
    
    # mae = np.mean(np.abs(real - fake_image.squeeze()))
    # psnr_val = psnr(real, fake_image, data_range=1.0)
    # ssim_val, _ = ssim(real, fake_image, full=True, data_range=1.0)
    
    # row = {
    #    'Patient_ID': subject_id,
    #    'MAE': mae,
    #    'PSNR': psnr_val,
    #    'SSIM': ssim_val,
    #}

    # Assuming `row` is a dictionary and `out_path` is a string path to the output directory
    # df = pd.DataFrame([row])
    #csv_path = os.path.join(out_path, "per_patient_metrics.csv")
    #write_header = not os.path.exists(csv_path)
    #df.to_csv(csv_path, mode='a', header=write_header, index=False)
    
    original_image = data['B'].as_tensor().permute(0, 2, 3, 4, 1)  # torch.from_numpy(data['B']).permute(0, 2, 3, 4, 1)
 
    complete_generated_image_3d = generated  # .squeeze(axis=(0, 4))
    
    original_image_3d = original_image.numpy().squeeze(axis=(0, 4))
    
    # Salva l'intero array tridimensionale come file .nii.gz
    complete_generated_image_nifti = nib.Nifti1Image(complete_generated_image_3d, np.eye(4))  # np.eye(4)
    nib.save(complete_generated_image_nifti, image_path_nifti)
    original_image_nifti = nib.Nifti1Image(original_image_3d, np.eye(4))  # np.eye(4)

    # original_image_nifti = resample_to_img(original_image_nifti, complete_generated_image_nifti, interpolation='linear')
    nib.save(original_image_nifti, image_path_original_nifti)

