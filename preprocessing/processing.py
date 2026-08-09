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

def visualize_volume(volume):
    """
    Enables fast scrolling through the volume slices using keyboard and mouse wheel.

    - Scroll wheel: Move forward/backward through slices
    - Up/Down arrows: Move through slices

    Parameters:
        volume (numpy.ndarray): The 3D CT volume (Z, Y, X).
    """
    fig, ax = plt.subplots()
    slice_index = volume.shape[2] // 2  # Assuming volume is Z, Y, X
    img_display = ax.imshow(volume[:, :, slice_index], cmap='gray', vmin=0, vmax=1)
    ax.set_title(f"Slice {slice_index}/{volume.shape[2]}")
    ax.axis("off")

    def update_slice(index):
        nonlocal slice_index
        slice_index = max(0, min(volume.shape[2] - 1, index))
        img_display.set_data(volume[:, :, slice_index])
        ax.set_title(f"Slice {slice_index}/{volume.shape[2]}")
        fig.canvas.draw_idle()

    def on_scroll(event):
        direction = 1 if event.step > 0 else -1
        update_slice(slice_index + direction)

    def on_key(event):
        if event.key in ['up', 'right']:
            update_slice(slice_index + 1)
        elif event.key in ['down', 'left']:
            update_slice(slice_index - 1)

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()

# ====== Lung Segmentation Function ======
def extract_lung_mask(ct_volume, threshold=-320, hole_area_thresh=1000, buffer_radius=5, min_region_area=1500):
    """
    Extracts a binary lung mask from a 3D CT scan using classic image processing.

    Parameters:
        ct_volume (np.ndarray): 3D CT volume (Y, X, Z) in Hounsfield Units.
        threshold (int): HU threshold to separate air from soft tissue.
        hole_area_thresh (int): Area threshold for hole filling per slice.
        buffer_radius (int): Radius (in pixels) for dilation to keep margin.
        min_region_area (int): Minimum area in pixels to consider a region as lung.

    Returns:
        np.ndarray: 3D binary lung mask with margin (same shape, dtype=uint8).
    """
    lung_mask = np.zeros_like(ct_volume, dtype=bool)
    z_min = int(ct_volume.shape[2] * 0.01)
    z_max = int(ct_volume.shape[2] * 0.99)

    for z in range(z_min, z_max):  # Skip extreme slices to avoid false positives
        slice_img = ct_volume[:, :, z]

        # Step 1: Threshold
        binary = slice_img < threshold

        # Step 2: Remove border-connected areas
        cleared = clear_border(binary)

        # Step 3: Label and filter regions
        label_img = measure.label(cleared)
        regions = [r for r in measure.regionprops(label_img) if r.area > min_region_area]
        regions = sorted(regions, key=lambda r: r.area, reverse=True)[:2]

        slice_mask = np.zeros_like(slice_img, dtype=bool)
        for r in regions:
            for coord in r.coords:
                slice_mask[coord[0], coord[1]] = True

        # Step 4: Fill small holes
        slice_mask = binary_closing(slice_mask, disk(3))
        slice_mask = morphology.remove_small_holes(slice_mask, area_threshold=hole_area_thresh)

        # Step 5: Dilate to keep margin
        slice_mask = binary_dilation(slice_mask, disk(buffer_radius))

        lung_mask[:, :, z] = slice_mask

    return lung_mask.astype(np.uint8)

# ====== Helpers ======
def load_nifti(path):
    nii = nib.load(path)
    return nii.get_fdata(), nii.affine, nii.header

def save_nifti(data, affine, header, path):
    nib.save(nib.Nifti1Image(data, affine, header), path)

def apply_mask(volume, mask):
    return volume * mask

def crop_to_mask(volume, mask):
    coords = np.array(np.nonzero(mask))
    min_coords = coords.min(axis=1)
    max_coords = coords.max(axis=1) + 1
    return volume[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1], min_coords[2]:max_coords[2]]

def has_overlap(mask1, mask2):
    return np.any(mask1 & mask2)

# ====== Utility Function for Padding or Cropping ======
def pad_to_match(reference, other):
    """
    Adjust 'other' to match the shape of 'reference' by padding or cropping.
    """
    ref_shape = np.array(reference.shape)
    other_shape = np.array(other.shape)

    if np.all(ref_shape == other_shape):
        return other

    result = other
    # Pad if smaller
    if np.any(ref_shape > other_shape):
        pad_width = [(0, max(0, r - o)) for r, o in zip(ref_shape, other_shape)]
        result = np.pad(result, pad_width, mode='constant', constant_values=0)

    # Crop if larger
    if np.any(ref_shape < other_shape):
        crop_slices = tuple(slice(0, r) for r in ref_shape)
        result = result[crop_slices]

    return result
    
# ====== Main Script ======
csv_path = "/mimer/NOBACKUP/groups/snic2022-5-277/rrestivo/Data/nifti_images/test.csv"  # train_whole_body_with_pathology.csv
ct_folder = "/mimer/NOBACKUP/groups/snic2022-5-277/piacente/IMMAGINI_DISTRETTI/test_lung_GAN_district"
output_folder = "/mimer/NOBACKUP/groups/snic2022-5-277/piacente/IMMAGINI_DISTRETTI/train_lung_GAN_district/test_set_segmented_lungs"
os.makedirs(output_folder, exist_ok=True)

df = pd.read_csv(csv_path)
results = []

for _, row in tqdm(df.iterrows()):
    subject_id = row["Subject ID"]
    diagnosis = str(row["diagnosis"]).lower()
    print(diagnosis)
    
    
    base = os.path.join(output_folder, subject_id)
    ct_out = f"{base}_ct_cropped.nii.gz"
    pet_orig_out = f"{base}_pet_original_cropped.nii.gz"
    pet_gen_out = f"{base}_pet_generated_cropped.nii.gz"

    #  Skip if all output files already exist
    if all(os.path.exists(p) for p in [ct_out, pet_orig_out, pet_gen_out]):
        print(f"⚠️ Skipping {subject_id} — output files already exist.")
        continue
    
    ct_files = [f for f in os.listdir(ct_folder) if f.startswith(subject_id) and f.endswith(".nii.gz") and "_CT_" in f]
    if not ct_files:
        print(f"CT not found for {subject_id}")
        continue

    ct_path = os.path.join(ct_folder, ct_files[0])
    ct_data, ct_affine, ct_header = load_nifti(ct_path)
    hu_min, hu_max = -1024, 3071
    ct_data = ct_data * (hu_max - hu_min) + hu_min
    print(ct_data.shape)
    # Extract unique number from the CT filename
    ct_filename = os.path.basename(ct_path)
    print(ct_filename)
    try:
        unique_number = ct_filename.split("_CT_")[1].replace(".nii.gz", "")
    except IndexError:
        print(f"❌ Could not parse unique number from {ct_filename}")
        continue

    # Match corresponding PETs and seg using that number
    pet_orig_path = os.path.join(ct_folder, f"{subject_id}_original_image_{unique_number}.nii.gz")
    pet_gen_path = os.path.join(ct_folder, f"{subject_id}_generated_image_{unique_number}.nii.gz")
    # seg_path = os.path.join(ct_folder, f"{unique_number}_SEG.nii.gz")
    seg_path = os.path.join(row["Path"], "SEG.nii.gz")

    if not all(os.path.exists(p) for p in [pet_orig_path, pet_gen_path]):
        print(f"Missing PET files for {subject_id}")
        continue

    pet_orig, _, _ = load_nifti(pet_orig_path)
    print(pet_orig.shape)
    pet_gen, _, _ = load_nifti(pet_gen_path)
    print(pet_gen.shape)

    lung_mask = extract_lung_mask(ct_data)
    print(lung_mask.shape)
    # visualize_volume(lung_mask)
    lesion_mask = None
    overlap_percent = 0.0

    if "negative" in diagnosis:
        lesion_mask = np.zeros_like(ct_data, dtype=bool)
    
    elif os.path.exists(seg_path):
        print(seg_path)
        seg_data, _, _ = load_nifti(seg_path)
        print(seg_data.shape)
        lesion_mask = seg_data > 0
        
        # Pad if needed
        if lung_mask.shape != lesion_mask.shape:
            lesion_mask = pad_to_match(lung_mask, lesion_mask)

        if not has_overlap(lung_mask, lesion_mask):
            print(f"No lesion in lungs for {subject_id}, skipping.")
            continue
        overlap_voxels = np.sum(lung_mask & lesion_mask)
        total_lesion_voxels = np.sum(lesion_mask)
        overlap_percent = (overlap_voxels / total_lesion_voxels) * 100 if total_lesion_voxels > 0 else 0.0
        print(f"Percentage of overlap: {overlap_percent}.")

    pet_orig_masked = apply_mask(pet_orig, lung_mask)
    pet_gen_masked = apply_mask(pet_gen, lung_mask)
    ct_masked = apply_mask(ct_data, lung_mask)
    

    try:
        ct_cropped = crop_to_mask(ct_masked, lung_mask)
        pet_orig_cropped = crop_to_mask(pet_orig_masked, lung_mask)
        pet_gen_cropped = crop_to_mask(pet_gen_masked, lung_mask)
    except ValueError as e:
        print(f"⚠️ Cropping failed for {subject_id}: {e}")
        continue
    

    #  base = os.path.join(output_folder, subject_id)
    # ct_out = f"{base}_ct_cropped.nii.gz"
    # pet_orig_out = f"{base}_pet_original_cropped.nii.gz"
    # pet_gen_out = f"{base}_pet_generated_cropped.nii.gz"

    save_nifti(ct_cropped, ct_affine, ct_header, ct_out)
    save_nifti(pet_orig_cropped, ct_affine, ct_header, pet_orig_out)
    save_nifti(pet_gen_cropped, ct_affine, ct_header, pet_gen_out)

    results.append({
        "Subject_ID": subject_id,
        "Diagnosis": diagnosis,
        "Overlap_Percent": round(overlap_percent, 2),
        "CT_Cropped_Path": ct_out,
        "PET_Original_Cropped_Path": pet_orig_out,
        "PET_Generated_Cropped_Path": pet_gen_out
    })
    print(f"✅ Processed {subject_id}")

    summary_csv_path = os.path.join(output_folder, "processing_summary.csv")
    pd.DataFrame(results).to_csv(summary_csv_path, index=False)
    print(f"\n📄 Summary CSV saved to: {summary_csv_path}")
