import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
from tqdm import tqdm

# === CONFIG ===
input_dir = '/mimer/NOBACKUP/groups/snic2022-5-277/piacente/IMMAGINI_TEST/WHOLE_BODY/lungs_only'  # folder with nii.gz files
output_ct_dir = '/mimer/NOBACKUP/groups/snic2022-5-277/piacente/IMMAGINI_TEST/WHOLE_BODY/lungs_only/slices/CT'
output_pet_dir = '/mimer/NOBACKUP/groups/snic2022-5-277/piacente/IMMAGINI_TEST/WHOLE_BODY/lungs_only/slices/PET'
csv_output_path = '/mimer/NOBACKUP/groups/snic2022-5-277/piacente/IMMAGINI_TEST/WHOLE_BODY/lungs_only/slices/ct_pet_slice_paths.csv'

# === SETUP ===
os.makedirs(output_ct_dir, exist_ok=True)
os.makedirs(output_pet_dir, exist_ok=True)

# === FIND AND MATCH FILES ===
ct_files = sorted([f for f in os.listdir(input_dir) if f.endswith('_ct_cropped.nii.gz')])
pet_files = sorted([f for f in os.listdir(input_dir) if f.endswith('_pet_original_cropped.nii.gz')])

def extract_id(filename):
    return filename.split('_')[1]

ct_dict = {extract_id(f): f for f in ct_files}
pet_dict = {extract_id(f): f for f in pet_files}
common_ids = sorted(set(ct_dict) & set(pet_dict))

print(f"Found {len(common_ids)} matching CT/PET volume pairs.")

rows = []

# === PROCESS EACH VOLUME PAIR ===
for pid in tqdm(common_ids):
    ct_path = os.path.join(input_dir, ct_dict[pid])
    pet_path = os.path.join(input_dir, pet_dict[pid])

    ct_volume = sitk.GetArrayFromImage(sitk.ReadImage(ct_path))  # shape: (slices, H, W)
    pet_volume = sitk.GetArrayFromImage(sitk.ReadImage(pet_path))

    if ct_volume.shape != pet_volume.shape:
        print(f"[Warning] Skipping {pid} due to shape mismatch: CT {ct_volume.shape}, PET {pet_volume.shape}")
        continue

    for i in range(ct_volume.shape[0]):
        ct_slice = ct_volume[i]
        pet_slice = pet_volume[i]

        ct_filename = f"{pid}_slice_{i:03d}_ct.npy"
        pet_filename = f"{pid}_slice_{i:03d}_pet.npy"

        ct_full_path = os.path.join(output_ct_dir, ct_filename)
        pet_full_path = os.path.join(output_pet_dir, pet_filename)

        np.save(ct_full_path, ct_slice)
        np.save(pet_full_path, pet_slice)

        rows.append({'CT_Path': ct_full_path, 'PET_Path': pet_full_path})

# === SAVE CSV ===
df = pd.DataFrame(rows)
df.to_csv(csv_output_path, index=False)

print(f"Saved {len(rows)} CT/PET slice pairs and CSV to:\n{csv_output_path}")