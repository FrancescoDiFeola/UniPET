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
from data.dataset_CACHE_Whole_body import CreateDataloader #data.dataset_CACHE ----- data.dataset_CACHE_Whole_body
#from data.dataset_CACHE import CreateDataloader
from monai.transforms import SpatialCropd
from monai.transforms import SpatialPadd
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def patch_indices(image_shape, patch_size=(32, 32, 32), overlap=(16, 16, 16)):
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


opt = TestOptions().parse()  # impostare phase == test

opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataloader(opt, shuffle=False, cache=False)
dataset_size = len(data_loader)
print('#testing images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# List to store MAE, PSNR, per paziente
mae_per_patient = []
mae_per_patient_list = []
std_mae_per_patient_list = []

psnr_per_patient = []
psnr_per_patient_list = []
std_psnr_per_patient_list = []

ssim_per_patient = []
ssim_per_patient_list = []
std_ssim_per_patient_list = []

# test
for i, data in enumerate(data_loader):
    # print(f"DATA: {data}")
    image_shape = data['A'].shape[2:]
    patch_centers = patch_indices(image_shape)
    # print(f"PATCH_CENTERS: {patch_centers}")
    print(f"IMAGE SHAPE: {image_shape}")

    mae_per_patient_patch = []
    psnr_per_patient_patch = []
    ssim_per_patient_patch = []

    for center in patch_centers:
        # print(f"CENTER: {center}")
        spatial_crop = SpatialCropd(keys=["A", "B"], roi_center=center, roi_size=[32, 32, 32])
        patches = spatial_crop(
            {"A": data['A'].as_tensor().permute(0, 2, 3, 4, 1), "B": data['B'].as_tensor().permute(0, 2, 3, 4, 1)})

        # print(f"PATCHES: {patches}")
        patches['A'] = patches['A'].squeeze().unsqueeze(0).unsqueeze(0)
        patches['B'] = patches['B'].squeeze().unsqueeze(0).unsqueeze(0)

        # model.set_input({key: value.squeeze() for key, value in patches.items()})
        model.set_input(patches)
        model.test()
        visuals = model.get_current_visuals()
        # img_path = model.get_image_paths()
        # print('process image... %s' % img_path)
        # visualizer.save_images(webpage, visuals, img_path)

        # Accesso a real_B e fake_B dal dizionario
        real_B_test = visuals['real_B']
        fake_B_test = visuals['fake_B']
        # Converti in tensori PyTorch float
        fake_B_test = torch.tensor(fake_B_test)
        real_B_test = torch.tensor(real_B_test)

        # Calcolo metriche
        mae_patch = np.mean(np.abs(real_B_test.numpy().squeeze() - fake_B_test.numpy().squeeze()))
        psnr_patch = psnr(real_B_test.numpy(), fake_B_test.numpy())
        ssim_patch, _ = ssim(real_B_test.numpy().squeeze(),fake_B_test.numpy().squeeze(),full=True,data_range=1.0)

        mae_per_patient_patch.append(mae_patch)  # Lista dei mae calcolati per ogni patch del singolo paziente
        psnr_per_patient_patch.append(psnr_patch)
        ssim_per_patient_patch.append(ssim_patch)

        mae_per_patient.append(mae_patch)  # Lista di tutti i mae di tutti i pazienti (calcolati sulle patch)
        psnr_per_patient.append(psnr_patch)
        ssim_per_patient.append(ssim_patch)

    mean_mae_per_patient = np.mean(mae_per_patient_patch)  # Media tra i mae calcolati sulle patch del paziente
    std_mae_per_patient = np.std(mae_per_patient_patch)
    mean_psnr_per_patient = np.mean(psnr_per_patient_patch)
    std_psnr_per_patient = np.std(psnr_per_patient_patch)
    mean_ssim_per_patient = np.mean(ssim_per_patient_patch)
    std_ssim_per_patient = np.std(ssim_per_patient_patch)

    mae_per_patient_list.append(mean_mae_per_patient)
    std_mae_per_patient_list.append(std_mae_per_patient)
    psnr_per_patient_list.append(mean_psnr_per_patient)
    std_psnr_per_patient_list.append(std_psnr_per_patient)
    ssim_per_patient_list.append(mean_ssim_per_patient)
    std_ssim_per_patient_list.append(std_ssim_per_patient)

    print(
        f'Progress: Paziente {i + 1}/{dataset_size} - MAE: {mean_mae_per_patient} ± {std_mae_per_patient} - PSNR: {mean_psnr_per_patient} ± {std_psnr_per_patient}'
        f' SSIM: {mean_ssim_per_patient} ± {std_ssim_per_patient}')

total_mean_mae = np.mean(mae_per_patient)
total_std_mae = np.std(mae_per_patient)
total_mean_psnr = np.mean(psnr_per_patient)
total_std_psnr = np.std(psnr_per_patient)
total_mean_ssim = np.mean(ssim_per_patient)
total_std_ssim = np.std(ssim_per_patient)

print("Mean MAE per paziente: ", mae_per_patient_list)
print("Deviazione standard per paziente MAE: ", std_mae_per_patient_list)
print("Mean PSNR per paziente: ", psnr_per_patient_list)
print("Deviazione standard per paziente PSNR: ", std_psnr_per_patient_list)
print("Mean SSIM per paziente: ", ssim_per_patient_list)
print("Deviazione standard per paziente SSIM: ", std_ssim_per_patient_list)

print("Mean MAE totale: ", total_mean_mae)
print("Deviazione standard totale MAE: ", total_std_mae)
print("Mean PSNR totale: ", total_mean_psnr)
print("Deviazione standard totale PSNR: ", total_std_psnr)
print("Mean SSIM totale: ", total_mean_ssim)
print("Deviazione standard totale SSIM: ", total_std_ssim)

webpage.save()
