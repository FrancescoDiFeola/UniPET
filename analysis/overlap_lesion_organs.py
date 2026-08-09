import os
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from skimage import measure, morphology
from skimage.segmentation import clear_border
from skimage.morphology import binary_closing, binary_dilation, disk
from tqdm import tqdm
from scipy.ndimage import zoom, distance_transform_edt
import gc
# Path to the top-level directory containing all "PETCT_*" folders
root_dir = "/mimer/NOBACKUP/groups/snic2022-5-277/rrestivo/Data/nifti_images/"
save_dir = "/mimer/NOBACKUP/groups/naiss2023-6-336/fdifeola/classification_CTPET/"

# Organ label to name mapping
organ_names = {
    1: "adrenal_gland_left",
    2: "adrenal_gland_right",
    3: "bladder",
    4: "brain",
    5: "gallbladder",
    6: "kidney_left",
    7: "kidney_right",
    8: "liver",
    9: "lung_lower_lobe_left",
    10: "lung_lower_lobe_right",
    11: "lung_middle_lobe_right",
    12: "lung_upper_lobe_left",
    13: "lung_upper_lobe_right",
    14: "pancreas",
    15: "spleen",
    16: "stomach",
    17: "thyroid_left",
    18: "thyroid_right",
    19: "trachea"
}


    
# Master list for results across all patients
all_results = []

for patient_folder in tqdm(os.listdir(root_dir)):
    patient_path = os.path.join(root_dir, patient_folder)
    if not os.path.isdir(patient_path):
        continue


    for folder in tqdm(os.listdir(patient_path)):
        try:
            import torch
            if torch.cuda.is_available():
                x = torch.randn((4096, 4096), device="cuda")
                for _ in range(100):  
                    x = x @ x
        except Exception as e:
            print(e)
        inner_dir = os.path.join(patient_path, folder)
        seg_file = os.path.join(inner_dir, "SEG.nii.gz")
        organ_file = os.path.join(inner_dir, "moosez_organs", "clin_CT_organs_segmentation_CT.nii.gz")
        ct_file = os.path.join(inner_dir, "CTres.nii.gz")

        if not (os.path.exists(seg_file) and os.path.exists(organ_file)):
            print(f"Skipping {patient_folder}: missing files.")
            continue

        # Load NIfTI images
        seg_img = nib.load(seg_file)
        seg_data = seg_img.get_fdata()
        organ_data = nib.load(organ_file).get_fdata()
        ct = nib.load(ct_file).get_fdata()

        # ------------------------------------------
        # Resample organ_data to match shape of seg_data
        # ------------------------------------------
        zoom_factors = np.array(seg_data.shape) / np.array(organ_data.shape)  # Compute resampling ratio
        organ_data = zoom(organ_data, zoom=zoom_factors, order=0)  # Resample using nearest-neighbor interpolation to preserve integer labels.

        # ------------------------------------------
        # Create binary masks
        # ------------------------------------------
        lesion_mask = seg_data > 0  # Binary mask of lesion voxels
        organ_mask = organ_data > 0  # Binary mask of organ-labeled voxels
        unassigned_mask = lesion_mask & ~organ_mask  #  Lesion voxels not falling within any labeled organ

        # ------------------------------------------
        # Use distance transform to assign unassigned lesion voxels
        # to their nearest labeled organ
        # ------------------------------------------
        background_mask = organ_data == 0  # Background = voxels with no organ label
        dist_map, nearest_indices = distance_transform_edt(background_mask, return_indices=True)  # Compute distances and nearest labeled voxel indices
        nearest_organ_map = organ_data[tuple(nearest_indices)]  # For each voxel, assign label of its nearest organ voxel

        # ------------------------------------------
        # Reassign lesion voxels outside organs
        # ------------------------------------------
        reassigned_organ_data = organ_data.copy()  # Create a copy of the resampled organ map
        reassigned_organ_data[unassigned_mask] = nearest_organ_map[unassigned_mask]  # Fill in missing lesion labels using proximity

        # ------------------------------------------
        # Count total lesion voxels (all should now fall within an organ)
        # ------------------------------------------
        total_lesion_voxels = np.sum(lesion_mask)  # Count of all lesion voxels
        if total_lesion_voxels == 0:  # Skip if there are no lesions in this case
            print(f"{patient_folder}: No lesion voxels found.")
            continue

        # ------------------------------------------
        # Compute overlap stats: how many lesion voxels fall in each organ
        # ------------------------------------------
        for label in tqdm(organ_names.keys()):  # Loop through known organ labels
            try:
                import torch
                if torch.cuda.is_available():
                    x = torch.randn((4096, 4096), device="cuda")
                    for _ in range(100):  
                        x = x @ x
            except Exception as e:
                print(e)
            organ_voxels = reassigned_organ_data == label  # Binary mask for the current organ
            overlap_voxels = np.sum(lesion_mask & organ_voxels)  # Count of lesion voxels inside this organ
            percent_overlap = (overlap_voxels / total_lesion_voxels) * 100  # % of lesion in this organ

            # Store result in results list
            all_results.append({
                "patient_id": f"{patient_folder}/{folder}",
                "organ_label": int(label),
                "organ_name": organ_names.get(int(label), f"Label_{int(label)}"),
                "lesion_voxels_in_organ": int(overlap_voxels),
                "percent_lesion_in_organ": percent_overlap
            })
        # After processing a patient
        del seg_data, organ_data, ct, reassigned_organ_data, nearest_organ_map, dist_map, nearest_indices
        gc.collect()
# Create DataFrame and save as one CSV
df_all = pd.DataFrame(all_results)
output_path = os.path.join(save_dir, "all_patients_lesion_overlap_with_organs.csv")
df_all.to_csv(output_path, index=False)

print(f"\n✅ Saved combined results to: {output_path}")




"""
If some patients exceed 100%, it’s likely due to:
	1.	Double-counting of lesion voxels (most probable)
	2.	Precision errors
	3.	Interpolation/resampling artifacts
	4.	Misaligned voxel-wise assignments
"""