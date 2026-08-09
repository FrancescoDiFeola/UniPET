import os
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from skimage import measure, morphology
from skimage.segmentation import clear_border
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, binary_dilation, disk
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


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



def extract_lung_mask_v2(ct_volume, hole_area_thresh=1000, buffer_radius=5, min_region_area=1500):
    lung_mask = np.zeros_like(ct_volume, dtype=bool)
    z_min = int(ct_volume.shape[2] * 0.01)
    z_max = int(ct_volume.shape[2] * 0.99)

    for z in range(z_min, z_max):
        slice_img = ct_volume[:, :, z]

        # Step 1: Otsu threshold (handle uniform slices with try/except)
        try:
            thresh = threshold_otsu(slice_img)
        except ValueError:
            continue  # Skip slice if Otsu fails (e.g., constant slice)

        binary = slice_img < thresh  # air & background will be True

        # Step 2: Clear border (external air)
        cleared = clear_border(binary)

        # Step 3: Label and region filter
        label_img = measure.label(cleared)
        filtered = np.zeros_like(slice_img, dtype=bool)
        img_height = slice_img.shape[0]

        for region in measure.regionprops(label_img):
            y, x = region.centroid
            if region.area > min_region_area and y < img_height * 0.85:
                for coord in region.coords:
                    filtered[coord[0], coord[1]] = True

        # Step 4: Clean up
        slice_mask = binary_closing(filtered, disk(3))
        slice_mask = morphology.remove_small_holes(slice_mask, area_threshold=hole_area_thresh)
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
csv_path = "/mimer/NOBACKUP/groups/snic2022-5-277/ct2pet/test_external.csv"  # train_whole_body_with_pathology.csv
ct_folder = "/mimer/NOBACKUP/groups/snic2022-5-277/piacente/test_external_generation_output"
output_folder = "/mimer/NOBACKUP/groups/snic2022-5-277/piacente/test_external_generation_output/lungs"
os.makedirs(output_folder, exist_ok=True)

df = pd.read_csv(csv_path)
results = []

for _, row in tqdm(df.iterrows()):
    full_path = row[0]
    subject_id = full_path.strip().split("/")[-1]

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

    if not all(os.path.exists(p) for p in [pet_orig_path, pet_gen_path]):
        print(f"Missing PET files for {subject_id}")
        continue

    pet_orig, _, _ = load_nifti(pet_orig_path)
    print(pet_orig.shape)
    pet_gen, _, _ = load_nifti(pet_gen_path)
    print(pet_gen.shape)

    lung_mask = extract_lung_mask_v2(ct_data)
    print(lung_mask.shape)

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
    print(pet_orig_cropped.shape, pet_gen_cropped.shape, ct_cropped.shape)
    mae = np.mean(np.abs(pet_orig_cropped - pet_gen_cropped))
    psnr_val = psnr(pet_orig_cropped, pet_gen_cropped, data_range=1.0)
    ssim_val, _ = ssim(pet_orig_cropped, pet_gen_cropped, full=True, data_range=1.0)

    row = {
        'Patient_ID': subject_id,
        'MAE': mae,
        'PSNR': psnr_val,
        'SSIM': ssim_val,
    }

    # Assuming `row` is a dictionary and `out_path` is a string path to the output directory
    df = pd.DataFrame([row])
    csv_path = os.path.join(output_folder, "per_patient_metrics.csv")
    write_header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode='a', header=write_header, index=False)

    save_nifti(ct_cropped, ct_affine, ct_header, ct_out)
    save_nifti(pet_orig_cropped, ct_affine, ct_header, pet_orig_out)
    save_nifti(pet_gen_cropped, ct_affine, ct_header, pet_gen_out)
