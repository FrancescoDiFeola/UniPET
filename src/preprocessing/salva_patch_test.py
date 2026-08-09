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
from data.dataset_ARMS import CreateDataloader
from monai.transforms import SpatialCropd
from monai.transforms import SpatialPadd
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import nibabel as nib  
import imageio  # Importa imageio per salvare le GIF

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

opt = TestOptions().parse() #impostare phase == test

opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataloader(opt, shuffle=False, cache=False)
dataset_size = len(data_loader)
print('#testing images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)

# Directory to save patches
output_dir = "/mimer/NOBACKUP/groups/snic2022-5-277/rrestivo/PATCHES_PIX2PIX/arms_new_2"
os.makedirs(output_dir, exist_ok=True)

# Set directories for visualizer
#visualizer.img_dir = os.path.join(output_dir, "images")
#visualizer.web_dir = os.path.join(output_dir, "web")
#os.makedirs(visualizer.img_dir, exist_ok=True)
#os.makedirs(visualizer.web_dir, exist_ok=True)


# Definisci la funzione per riscalare un'immagine nell'intervallo 0-255
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

# Funzione per salvare un'immagine 3D come GIF
def save_gif(image_3d, path, duration=0.1):
    frames = [image_3d[:, :, i] for i in range(image_3d.shape[2])]
    imageio.mimsave(path, frames, duration=duration)

# Test
for i, data in enumerate(data_loader):
    if i >= 3:  # Processa solo i primi 5 pazienti
        break
    
    image_shape = data['A'].shape[2:]
    patch_centers = patch_indices(image_shape)
    print(f"Elaborazione del paziente {i+1} con forma dell'immagine: {image_shape}")

    for j, center in enumerate(patch_centers):  # Assicura che `j` sia definito qui
        spatial_crop = SpatialCropd(keys=["A", "B"], roi_center=center, roi_size=[32, 32, 32])
        patches = spatial_crop({"A": data['A'].as_tensor().permute(0, 2, 3, 4, 1), "B": data['B'].as_tensor().permute(0, 2, 3, 4, 1)})
        
        patches['A'] = patches['A'].squeeze().unsqueeze(0).unsqueeze(0)
        patches['B'] = patches['B'].squeeze().unsqueeze(0).unsqueeze(0)

        model.set_input(patches)
        model.test()
        visuals = model.get_current_visuals()

        real_B_test = torch.tensor(visuals['real_B']).squeeze().numpy()
        fake_B_test = torch.tensor(visuals['fake_B']).squeeze().numpy()

        # Riscala le immagini nell'intervallo 0-255
        real_B_test_scaled = scale_image(real_B_test)
        fake_B_test_scaled = scale_image(fake_B_test)
        #fake_B_test_scaled = scale_fake_with_original(fake_B_test, real_B_test)

        #Gaussian Blur
        blurred_image_3d = gaussian_filter(fake_B_test_scaled, sigma=1)

        # Salva le patch come file NIfTI
        real_B_nifti = nib.Nifti1Image(blurred_image_3d, np.eye(4))
        fake_B_nifti = nib.Nifti1Image(fake_B_test_scaled, np.eye(4))

        real_B_path = os.path.join(output_dir, f'patient_{i+1}_patch_{j+1}_real_B.nii.gz')
        fake_B_path = os.path.join(output_dir, f'patient_{i+1}_patch_{j+1}_fake_B.nii.gz')

        #nib.save(real_B_nifti, real_B_path)
        #nib.save(fake_B_nifti, fake_B_path)

        # Salva le patch come GIF
        real_B_gif_path = os.path.join(output_dir, f'patient_{i+1}_patch_{j+1}_real_B.gif')
        fake_B_gif_path = os.path.join(output_dir, f'patient_{i+1}_patch_{j+1}_fake_B.gif')

        save_gif(real_B_test_scaled, real_B_gif_path)
        save_gif(blurred_image_3d, fake_B_gif_path)

        print(f"Salvata patch {j+1} per il paziente {i+1}")


print("Finished processing and saving patches.")


