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

# ==== Parameters ====
patch_dim = 32
overlap_dim = 16


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


opt = TestOptions().parse()  # impostare phase == test
# ==== Device ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataloader(opt, shuffle=False, cache=False)  # CreateDataloader_TEST
dataset_size = len(data_loader)
print('#testing images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)

csv_path = "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/curriculumGAN_test_whole_body.csv"

mae_per_patient = []
psnr_per_patient = []
ssim_per_patient = []
all_results = []
# test
for i, data in enumerate(data_loader):

    A, B = data["A"], data["B"]
    shape = A.shape[2:]
    patches = patch_indices(shape)
    _, _, D, H, W = A.shape

    generated = np.zeros((D, H, W), dtype=np.float32)
    overlap = np.zeros((D, H, W), dtype=np.float32)

    for center in patches:
        crop = SpatialCropd(keys=["A", "B"], roi_center=center, roi_size=[patch_dim] * 3)
        patches = crop({
            "A": A.as_tensor().permute(0, 2, 3, 4, 1),
            "B": B.as_tensor().permute(0, 2, 3, 4, 1)
        })
        patches['A'] = patches['A'].squeeze().unsqueeze(0).unsqueeze(0).to(device)
        patches['B'] = patches['B'].squeeze().unsqueeze(0).unsqueeze(0).to(device)

        model.set_input(patches)
        model.test()
        visuals = model.get_current_visuals()

        real_B_test = visuals['real_B']
        fake_B_test = visuals['fake_B'].squeeze().cpu().numpy()

        cx, cy, cz = center
        sx = sy = sz = overlap_dim
        generated[cx - sx:cx + sx, cy - sy:cy + sy, cz - sz:cz + sz] += fake_B_test
        overlap[cx - sx:cx + sx, cy - sy:cy + sy, cz - sz:cz + sz] += 1

    generated /= np.maximum(overlap, 1)
    fake_image = generated

    # ==== Metrics ====
    real = denorm(B).squeeze().numpy()
    fake = denorm(torch.tensor(generated)).numpy()
    print(f"shapes: {real.shape}, {fake.shape}")
    print(f"shapes: {real.shape}, {fake.shape}")
    # Calcolo metriche
    mae = np.mean(np.abs(real.cpu() - fake_image.squeeze()))
    psnr_val = psnr(real, fake, data_range=20.0)
    ssim_val, _ = ssim(real, fake, full=True, data_range=20.0)

    real_metrics = compute_pet_metrics(real.squeeze())
    fake_metrics = compute_pet_metrics(fake.squeeze())

    print(f'Progress: Paziente {i + 1}/{dataset_size} - MAE: {mae_value} - PSNR: {psnr_value} - SSIM: {ssim_value}')

    row = {
        'Patient_ID': data['B_paths'],
        'MAE': mae,
        'PSNR': psnr_val,
        'SSIM': ssim_val,
        'REAL_SUV_max': real_metrics['SUV_max'],
        'REAL_SUV_mean': real_metrics['SUV_mean'],
        'REAL_MTV_1.5': real_metrics['MTV_1.5'],
        'REAL_TLG_1.5': real_metrics['TLG_1.5'],
        'REAL_MTV_2.5': real_metrics['MTV_2.5'],
        'REAL_TLG_2.5': real_metrics['TLG_2.5'],
        'GEN_SUV_max': fake_metrics['SUV_max'],
        'GEN_SUV_mean': fake_metrics['SUV_mean'],
        'GEN_MTV_1.5': fake_metrics['MTV_1.5'],
        'GEN_TLG_1.5': fake_metrics['TLG_1.5'],
        'GEN_MTV_2.5': fake_metrics['MTV_2.5'],
        'GEN_TLG_2.5': fake_metrics['TLG_2.5'],
    }
    df = pd.DataFrame([row])
    write_header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode='a', header=write_header, index=False)
    print(f"[INFO] Saved Patient {i + 1} metrics to: {csv_path}")

    # Salva l'immagine completa generata nella cartella out_path
    # out_path = "/mimer/NOBACKUP/groups/snic2022-5-277/rrestivo/IMMAGINI_PIX2PIX/arms"
    out_path = opt.out_path  # mimer/NOBACKUP/groups/snic2022-5-277/piacente/IMMAGINI_TEST
    os.makedirs(out_path, exist_ok=True)

    image_filename = f"generated_image_{i + 1}.png"  # Puoi personalizzare il nome del file come desideri
    image_path = os.path.join(out_path, image_filename)
    image_filename_nifti = f"generated_image_{i + 1}.nii.gz"  # Puoi personalizzare il nome del file come desideri
    image_path_nifti = os.path.join(out_path, image_filename_nifti)
    image_filename_original = f"original_image_{i + 1}.png"  # Puoi personalizzare il nome del file come desideri
    image_path_original = os.path.join(out_path, image_filename_original)
    image_filename_original_nifti = f"original_image_{i + 1}.nii.gz"  # Puoi personalizzare il nome del file come desideri
    image_path_original_nifti = os.path.join(out_path, image_filename_original_nifti)
    image_filename_CT = f"{data['A_paths']}_original_CT_{i + 1}.nii.gz"
    image_path_original_CT_nifti = os.path.join(out_path, image_filename_CT)

    # Estrai l'immagine tridimensionale dalla forma (1, 269, 295, 83, 1)
    original_image = data['B'].as_tensor().permute(0, 2, 3, 4, 1)
    original_CT = data['A'].as_tensor().permute(0, 2, 3, 4, 1)

    original_CT_3d = original_CT.numpy().squeeze(axis=(0, 4))

    # Salva l'intero array tridimensionale come file .nii.gz
    
    # complete_generated_image_nifti = nib.Nifti1Image(generated, np.eye(4))
    # nib.save(complete_generated_image_nifti, image_path_nifti)
    # original_image_nifti = nib.Nifti1Image(real, np.eye(4))
    # nib.save(original_image_nifti, image_path_original_nifti)

    # original_CT_nifti = nib.Nifti1Image(original_CT_3d, np.eye(4))
    # nib.save(original_CT_nifti, image_path_original_CT_nifti)

# ==== Average Metrics ====
# df_all = pd.read_csv(csv_path)
# df_avg = df_all.mean(numeric_only=True).to_frame().T
# df_avg.to_csv(os.path.join(results_dir, "curriculumGAN_test_metrics.csv"), index=False)
# print(f"[INFO] Saved average metrics")
