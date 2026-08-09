import os
import time
import datetime
import torch
import numpy as np
import pandas as pd
from monai.transforms import SpatialCropd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from options.test_options import TestOptions
from models.models import create_model
from util.visualizer import Visualizer
from data.dataset_CACHE import CreateDataloader  # from data.dataset_CACHE_Whole_body import CreateDataloader

# =============================
# ==== Parameters & Setup =====
# =============================
PATCH_DIM = 32
OVERLAP_DIM = 16

def patch_indices(shape, patch=PATCH_DIM, overlap=OVERLAP_DIM):
    """Generate 3D patch center indices for sliding window inference."""
    indices, step = [], patch - overlap
    for x in range(0, shape[0], step):
        for y in range(0, shape[1], step):
            for z in range(0, shape[2], step):
                cx = min(x + patch // 2, shape[0] - overlap)
                cy = min(y + patch // 2, shape[1] - overlap)
                cz = min(z + patch // 2, shape[2] - overlap)
                indices.append((cx, cy, cz))
    return indices


def denorm(tensor, a_min=0, a_max=20):
    """Denormalize tensor scaled to [0,1] back to physical range."""
    return tensor * (a_max - a_min) + a_min


def compute_pet_metrics(suv_values):
    """Compute standard PET quantitative metrics."""
    suv_values = suv_values.flatten()
    suv_max = np.max(suv_values)
    suv_mean = np.mean(suv_values)
    mtv_15 = np.sum(suv_values >= 1.5)
    mtv_25 = np.sum(suv_values >= 2.5)
    return {
        'SUV_max': suv_max,
        'SUV_mean': suv_mean,
        'MTV_1.5': mtv_15,
        'TLG_1.5': mtv_15 * suv_mean,
        'MTV_2.5': mtv_25,
        'TLG_2.5': mtv_25 * suv_mean
    }


# =============================
# ==== Initialization =========
# =============================
opt = TestOptions().parse()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

opt.nThreads = 1
opt.batchSize = 1
opt.serial_batches = True
opt.no_flip = True

data_loader = CreateDataloader(opt, shuffle=False, cache=False)
dataset_size = len(data_loader)
print(f"# Testing images: {dataset_size}")

model = create_model(opt)
visualizer = Visualizer(opt)

# ==== CSV output ====
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola"
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, f"random_ext_test_{opt.test_district}.csv")

# =============================
# ==== Inference Loop =========
# =============================
for i, data in enumerate(data_loader, start=1):

    A, B = data["A"], data["B"]
    shape = A.shape[2:]
    patch_centers = patch_indices(shape)
    _, _, D, H, W = A.shape

    generated = np.zeros((D, H, W), dtype=np.float32)
    overlap = np.zeros((D, H, W), dtype=np.float32)

    for center in patch_centers:
        crop = SpatialCropd(keys=["A", "B"], roi_center=center, roi_size=[PATCH_DIM] * 3)
        patch_dict = crop({
            "A": A.as_tensor().permute(0, 2, 3, 4, 1),
            "B": B.as_tensor().permute(0, 2, 3, 4, 1)
        })
        patch_dict['A'] = patch_dict['A'].squeeze().unsqueeze(0).unsqueeze(0).to(device)
        patch_dict['B'] = patch_dict['B'].squeeze().unsqueeze(0).unsqueeze(0).to(device)

        model.set_input(patch_dict)
        model.test()
        visuals = model.get_current_visuals()

        fake_B = visuals['fake_B']
        fake_B_test = fake_B.squeeze().cpu().numpy() if isinstance(fake_B, torch.Tensor) else np.squeeze(fake_B)

        cx, cy, cz = center
        sx = sy = sz = OVERLAP_DIM
        generated[cx - sx:cx + sx, cy - sy:cy + sy, cz - sz:cz + sz] += fake_B_test
        overlap[cx - sx:cx + sx, cy - sy:cy + sy, cz - sz:cz + sz] += 1

    generated /= np.maximum(overlap, 1)

    # ==== Metrics ====
    real = denorm(B).squeeze().cpu().numpy()
    fake = denorm(torch.tensor(generated)).numpy()

    mae = np.mean(np.abs(real - fake))
    psnr_val = psnr(real, fake, data_range=20.0)
    ssim_val, _ = ssim(real, fake, full=True, data_range=20.0)

    real_metrics = compute_pet_metrics(real)
    fake_metrics = compute_pet_metrics(fake)

    patient_id = os.path.basename(data['B_paths'][0]) if isinstance(data['B_paths'], list) else str(data['B_paths'])
    print(f"[{i}/{dataset_size}] {patient_id} | MAE: {mae:.4f} | PSNR: {psnr_val:.2f} | SSIM: {ssim_val:.3f}")

    # ==== Save to CSV ====
    row = {
        'Patient_ID': data['B_paths'],
        'MAE': mae,
        'PSNR': psnr_val,
        'SSIM': ssim_val,
        **{f'REAL_{k}': v for k, v in real_metrics.items()},
        **{f'GEN_{k}': v for k, v in fake_metrics.items()}
    }

    df = pd.DataFrame([row])
    write_header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode='a', header=write_header, index=False)

    print(f"   ✅ Saved metrics to: {csv_path}")

print("\n=== Inference Completed ===")
print(f"Results saved to: {csv_path}")