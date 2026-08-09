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
import re
import nibabel as nib
import matplotlib.pyplot as plt
#from data.dataset_TEST import CreateDataloader_TEST
from data.dataset_CACHE_Whole_body import CreateDataloader #data.dataset_CACHE
# from data.dataset_CACHE import CreateDataloader
from monai.transforms import SpatialCropd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import imageio
from scipy.ndimage import gaussian_filter
import pandas as pd
from tqdm import tqdm

patch_dim = 32
overlap_dim = 16
def patch_indices(image_shape, patch_size=(patch_dim, patch_dim, patch_dim), overlap=(overlap_dim, overlap_dim, overlap_dim)): # Calcola i centri dei patch 3D per suddividere un'immagine in porzioni piu piccole con sovrapposizione
    indices = []
    step_size = np.subtract(patch_size, overlap)

    for x in range(0, image_shape[0], step_size[0]):
        for y in range(0, image_shape[1], step_size[1]):
            for z in range(0, image_shape[2], step_size[2]):
                center = (x + patch_size[0] // 2, y + patch_size[1] // 2, z + patch_size[2] // 2)

                # Assicurati che il centro non superi le dimensioni dell'immagine
                center = (
                    min(center[0], image_shape[0] - overlap[0]),
                    min(center[1], image_shape[1] - overlap[1]),
                    min(center[2], image_shape[2] - overlap[2])
                )

                indices.append(center)

    return indices

def save_gif(image_3d, path, duration=0.1): # Salva una sequenza di immagini 2D come una GIF animata.
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


opt = TestOptions().parse()  # impostare phase == test

opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip


data_loader = CreateDataloader(opt, shuffle=False, cache=False) # CreateDataloader_TEST
dataset_size = len(data_loader)
print('#testing images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)

mae_per_patient = []
psnr_per_patient = []
ssim_per_patient = []
# test
for i, data in tqdm(enumerate(data_loader)):
    # print(f"DATA: {data}")
    image_shape = data['A'].shape[2:]
    original_CT = data['A'].as_tensor().permute(0, 2, 3, 4, 1)
    original_CT_3d = original_CT.numpy().squeeze(axis=(0, 4))
    out_path = '/mimer/NOBACKUP/groups/snic2022-5-277/piacente/IMMAGINI_TEST/WHOLE_BODY/training_set'    # 'mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/CTPET_train'
    os.makedirs(out_path, exist_ok=True)
    match = re.search(r'(PETCT_\w{10})', data["A_paths"][0])
    subject_id = match.group(1) if match else f"unknown_{i+1}"
    image_filename_CT = f"{subject_id}_original_CT_{i+1}.nii.gz"
    image_path_original_CT_nifti = os.path.join(out_path, image_filename_CT)
    original_CT_nifti = nib.Nifti1Image(original_CT_3d, np.eye(4))
    nib.save(original_CT_nifti, image_path_original_CT_nifti)
    
    patch_centers = patch_indices(image_shape)
    #print(f"DATA_shape: {image_shape}")
    #print(f"Data shape dopo permute: {data['B'].as_tensor().permute(0, 2, 3, 4, 1).shape}")

    # Inizializza un array numpy per l'immagine completa generata
    complete_generated_image = np.zeros_like(data['B'].as_tensor().permute(0, 2, 3, 4, 1))
    # Inizializza contatore per tener traccia del numero di patch sovrapposte in ciascun punto
    overlap_counter = np.zeros_like(data['B'].as_tensor().permute(0, 2, 3, 4, 1))

    for center in tqdm(patch_centers):
        #print(f"CENTER: {center}")
        spatial_crop = SpatialCropd(keys=["A", "B"], roi_center=center, roi_size=[patch_dim, patch_dim, patch_dim])
        patches = spatial_crop({"A": data['A'].as_tensor().permute(0, 2, 3, 4, 1), "B": data['B'].as_tensor().permute(0, 2, 3, 4, 1)})
        # print(f"PATCHES: {patches}")
        patches['A'] = patches['A'].squeeze().unsqueeze(0).unsqueeze(0)
        patches['B'] = patches['B'].squeeze().unsqueeze(0).unsqueeze(0)

        model.set_input(patches)
        model.test()
        visuals = model.get_current_visuals() # restituisce un dizionario di immagini: real_A, fake_B, real_B
        # img_path = model.get_image_paths()
        # print('process image... %s' % img_path)
        # visualizer.save_images(webpage, visuals, img_path)

        # Accesso a real_B e fake_B dal dizionario
        real_B_test = visuals['real_B']
        fake_B_test = visuals['fake_B']


        complete_generated_image[0,
            center[0] - overlap_dim:center[0] + overlap_dim,
            center[1] - overlap_dim:center[1] + overlap_dim,
            center[2] - overlap_dim:center[2] + overlap_dim,
            0] += fake_B_test.squeeze()
        #print(complete_generated_image)

        overlap_counter[0,
            center[0] - overlap_dim:center[0] + overlap_dim,
            center[1] - overlap_dim:center[1] + overlap_dim,
            center[2] - overlap_dim:center[2] + overlap_dim,
            0] += 1
        #print(overlap_counter)


        # Converti in tensori PyTorch
        fake_B_test = torch.tensor(fake_B_test)
        real_B_test = torch.tensor(real_B_test)


    # RICOSTRUZIONE IMMAGINE: Calcola la media finale dividendo per il numero di patch sovrapposte
    complete_generated_image /= np.maximum(overlap_counter, 1)
    
    #print("REAL IMAGE", data['A'].as_tensor().permute(0, 2, 3, 4, 1).numpy())
    #print("OVERLAP COUNTER", overlap_counter)
    #print("GENERATED IMAGE", complete_generated_image)



    image_filename_nifti = f"{subject_id}_generated_image_{i + 1}.nii.gz"  # Puoi personalizzare il nome del file come desideri
    image_path_nifti = os.path.join(out_path, image_filename_nifti)
    image_filename_original_nifti = f"{subject_id}_original_image_{i + 1}.nii.gz"  # Puoi personalizzare il nome del file come desideri
    image_path_original_nifti = os.path.join(out_path, image_filename_original_nifti)

    
    # Normalizza l'immagine nell'intervallo [0, 1] prima di salvarla
    # complete_generated_image_normalized = (complete_generated_image - np.min(complete_generated_image)) / (np.max(complete_generated_image) - np.min(complete_generated_image))

    
    # Estrai l'immagine tridimensionale dalla forma (1, 269, 295, 83, 1)
    original_image = data['B'].as_tensor().permute(0, 2, 3, 4, 1)

    # Estrai il valore massimo e minimo da original_image
    # valore_massimo = np.max(original_image.numpy())
    # valore_minimo = np.min(original_image.numpy())
    # original_image_slice = original_image.numpy()[:, 30, :, :, :].squeeze()
    # complete_generated_image_slice = complete_generated_image[:, 30, :, :, :].squeeze()
    # Effettua il windowing di complete_generated_image
    # complete_generated_image_windowed = np.clip(complete_generated_image, valore_minimo, valore_massimo)

    # Estrai l'immagine tridimensionale dalla forma (1, 269, 295, 83, 1)
    # complete_generated_image_3d = complete_generated_image_windowed.squeeze(axis=(0, 4))
    
    complete_generated_image_3d = complete_generated_image.squeeze(axis=(0, 4))
    original_image_3d = original_image.numpy().squeeze(axis=(0, 4))
    

    # Salva l'intero array tridimensionale come file .nii.gz
    complete_generated_image_nifti = nib.Nifti1Image(complete_generated_image_3d, np.eye(4))
    nib.save(complete_generated_image_nifti, image_path_nifti)
    original_image_nifti = nib.Nifti1Image(original_image_3d, np.eye(4))
    nib.save(original_image_nifti, image_path_original_nifti)
    

    # Salva l'immagine utilizzando matplotlib
    # plt.imsave(image_path, complete_generated_image_slice, cmap='gray')
    # plt.imsave(image_path_original, original_image_slice, cmap='gray')

    # Riscala le immagini nell'intervallo 0-255
    # nifti_scaled = scale_image(complete_generated_image_3d)
    #nifti_scaled = scale_fake_with_original(complete_generated_image_3d, original_image_3d)
    # blurred_image_3d = gaussian_filter(nifti_scaled, sigma=1) #gaussian blur
    # real_scaled = scale_image(original_image_3d)

    # Salva le patch come GIF
    # real_B_gif_path = os.path.join(out_path, f'patient_{i+1}_real.gif')
    # fake_B_gif_path = os.path.join(out_path, f'patient_{i+1}_fake.gif')

    # save_gif(blurred_image_3d, fake_B_gif_path)
    # save_gif(real_scaled, real_B_gif_path)
