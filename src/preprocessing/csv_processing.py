import pandas as pd
import nibabel as nib

# Load CSV
csv_path = '/mimer/NOBACKUP/groups/snic2022-5-277/piacente/my_approach/fold_0.csv'
df = pd.read_csv(csv_path)

# Function to get volume shape from a NIfTI file
def get_nifti_shape(path):
    try:
        img = nib.load(path)
        return img.shape
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return (None, None, None)

# Apply the shape extraction to each path column
df[['CT_shape_x', 'CT_shape_y', 'CT_shape_z']] = df['CT_Cropped_Path'].apply(lambda x: pd.Series(get_nifti_shape(x)))
df[['PET_original_shape_x', 'PET_original_shape_y', 'PET_original_shape_z']] = df['PET_Original_Cropped_Path'].apply(lambda x: pd.Series(get_nifti_shape(x)))
df[['PET_generated_shape_x', 'PET_generated_shape_y', 'PET_generated_shape_z']] = df['PET_Generated_Cropped_Path'].apply(lambda x: pd.Series(get_nifti_shape(x)))

# Save the updated CSV
df.to_csv('/mimer/NOBACKUP/groups/snic2022-5-277/piacente/my_approach/fold_0_with_shapes.csv', index=False)

# Load CSV
csv_path = '/mimer/NOBACKUP/groups/snic2022-5-277/piacente/my_approach/fold_1.csv'
df = pd.read_csv(csv_path)

# Function to get volume shape from a NIfTI file
def get_nifti_shape(path):
    try:
        img = nib.load(path)
        return img.shape
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return (None, None, None)

# Apply the shape extraction to each path column
df[['CT_shape_x', 'CT_shape_y', 'CT_shape_z']] = df['CT_Cropped_Path'].apply(lambda x: pd.Series(get_nifti_shape(x)))
df[['PET_original_shape_x', 'PET_original_shape_y', 'PET_original_shape_z']] = df['PET_Original_Cropped_Path'].apply(lambda x: pd.Series(get_nifti_shape(x)))
df[['PET_generated_shape_x', 'PET_generated_shape_y', 'PET_generated_shape_z']] = df['PET_Generated_Cropped_Path'].apply(lambda x: pd.Series(get_nifti_shape(x)))

# Save the updated CSV
df.to_csv('/mimer/NOBACKUP/groups/snic2022-5-277/piacente/my_approach/fold_1_with_shapes.csv', index=False)

# Load CSV
csv_path = '/mimer/NOBACKUP/groups/snic2022-5-277/piacente/my_approach/fold_2.csv'
df = pd.read_csv(csv_path)

# Function to get volume shape from a NIfTI file
def get_nifti_shape(path):
    try:
        img = nib.load(path)
        return img.shape
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return (None, None, None)

# Apply the shape extraction to each path column
df[['CT_shape_x', 'CT_shape_y', 'CT_shape_z']] = df['CT_Cropped_Path'].apply(lambda x: pd.Series(get_nifti_shape(x)))
df[['PET_original_shape_x', 'PET_original_shape_y', 'PET_original_shape_z']] = df['PET_Original_Cropped_Path'].apply(lambda x: pd.Series(get_nifti_shape(x)))
df[['PET_generated_shape_x', 'PET_generated_shape_y', 'PET_generated_shape_z']] = df['PET_Generated_Cropped_Path'].apply(lambda x: pd.Series(get_nifti_shape(x)))

# Save the updated CSV
df.to_csv('/mimer/NOBACKUP/groups/snic2022-5-277/piacente/my_approach/fold_2_with_shapes.csv', index=False)

# Load CSV
csv_path = '/mimer/NOBACKUP/groups/snic2022-5-277/piacente/my_approach/fold_3.csv'
df = pd.read_csv(csv_path)

# Function to get volume shape from a NIfTI file
def get_nifti_shape(path):
    try:
        img = nib.load(path)
        return img.shape
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return (None, None, None)

# Apply the shape extraction to each path column
df[['CT_shape_x', 'CT_shape_y', 'CT_shape_z']] = df['CT_Cropped_Path'].apply(lambda x: pd.Series(get_nifti_shape(x)))
df[['PET_original_shape_x', 'PET_original_shape_y', 'PET_original_shape_z']] = df['PET_Original_Cropped_Path'].apply(lambda x: pd.Series(get_nifti_shape(x)))
df[['PET_generated_shape_x', 'PET_generated_shape_y', 'PET_generated_shape_z']] = df['PET_Generated_Cropped_Path'].apply(lambda x: pd.Series(get_nifti_shape(x)))

# Save the updated CSV
df.to_csv('/mimer/NOBACKUP/groups/snic2022-5-277/piacente/my_approach/fold_3_with_shapes.csv', index=False)

# Load CSV
csv_path = '/mimer/NOBACKUP/groups/snic2022-5-277/piacente/my_approach/fold_4.csv'
df = pd.read_csv(csv_path)

# Function to get volume shape from a NIfTI file
def get_nifti_shape(path):
    try:
        img = nib.load(path)
        return img.shape
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return (None, None, None)

# Apply the shape extraction to each path column
df[['CT_shape_x', 'CT_shape_y', 'CT_shape_z']] = df['CT_Cropped_Path'].apply(lambda x: pd.Series(get_nifti_shape(x)))
df[['PET_original_shape_x', 'PET_original_shape_y', 'PET_original_shape_z']] = df['PET_Original_Cropped_Path'].apply(lambda x: pd.Series(get_nifti_shape(x)))
df[['PET_generated_shape_x', 'PET_generated_shape_y', 'PET_generated_shape_z']] = df['PET_Generated_Cropped_Path'].apply(lambda x: pd.Series(get_nifti_shape(x)))

# Save the updated CSV
df.to_csv('/mimer/NOBACKUP/groups/snic2022-5-277/piacente/my_approach/fold_4_with_shapes.csv', index=False)

# Load CSV
csv_path = '/mimer/NOBACKUP/groups/snic2022-5-277/piacente/my_approach/fold_5.csv'
df = pd.read_csv(csv_path)

# Function to get volume shape from a NIfTI file
def get_nifti_shape(path):
    try:
        img = nib.load(path)
        return img.shape
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return (None, None, None)

# Apply the shape extraction to each path column
df[['CT_shape_x', 'CT_shape_y', 'CT_shape_z']] = df['CT_Cropped_Path'].apply(lambda x: pd.Series(get_nifti_shape(x)))
df[['PET_original_shape_x', 'PET_original_shape_y', 'PET_original_shape_z']] = df['PET_Original_Cropped_Path'].apply(lambda x: pd.Series(get_nifti_shape(x)))
df[['PET_generated_shape_x', 'PET_generated_shape_y', 'PET_generated_shape_z']] = df['PET_Generated_Cropped_Path'].apply(lambda x: pd.Series(get_nifti_shape(x)))

# Save the updated CSV
df.to_csv('/mimer/NOBACKUP/groups/snic2022-5-277/piacente/my_approach/fold_5_with_shapes.csv', index=False)

# Load CSV
csv_path = '/mimer/NOBACKUP/groups/snic2022-5-277/piacente/my_approach/fold_6.csv'
df = pd.read_csv(csv_path)

# Function to get volume shape from a NIfTI file
def get_nifti_shape(path):
    try:
        img = nib.load(path)
        return img.shape
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return (None, None, None)

# Apply the shape extraction to each path column
df[['CT_shape_x', 'CT_shape_y', 'CT_shape_z']] = df['CT_Cropped_Path'].apply(lambda x: pd.Series(get_nifti_shape(x)))
df[['PET_original_shape_x', 'PET_original_shape_y', 'PET_original_shape_z']] = df['PET_Original_Cropped_Path'].apply(lambda x: pd.Series(get_nifti_shape(x)))
df[['PET_generated_shape_x', 'PET_generated_shape_y', 'PET_generated_shape_z']] = df['PET_Generated_Cropped_Path'].apply(lambda x: pd.Series(get_nifti_shape(x)))

# Save the updated CSV
df.to_csv('/mimer/NOBACKUP/groups/snic2022-5-277/piacente/my_approach/fold_6_with_shapes.csv', index=False)

# Load CSV
csv_path = '/mimer/NOBACKUP/groups/snic2022-5-277/piacente/my_approach/fold_7.csv'
df = pd.read_csv(csv_path)

# Function to get volume shape from a NIfTI file
def get_nifti_shape(path):
    try:
        img = nib.load(path)
        return img.shape
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return (None, None, None)

# Apply the shape extraction to each path column
df[['CT_shape_x', 'CT_shape_y', 'CT_shape_z']] = df['CT_Cropped_Path'].apply(lambda x: pd.Series(get_nifti_shape(x)))
df[['PET_original_shape_x', 'PET_original_shape_y', 'PET_original_shape_z']] = df['PET_Original_Cropped_Path'].apply(lambda x: pd.Series(get_nifti_shape(x)))
df[['PET_generated_shape_x', 'PET_generated_shape_y', 'PET_generated_shape_z']] = df['PET_Generated_Cropped_Path'].apply(lambda x: pd.Series(get_nifti_shape(x)))

# Save the updated CSV
df.to_csv('/mimer/NOBACKUP/groups/snic2022-5-277/piacente/my_approach/fold_7_with_shapes.csv', index=False)

# Load CSV
csv_path = '/mimer/NOBACKUP/groups/snic2022-5-277/piacente/my_approach/fold_8.csv'
df = pd.read_csv(csv_path)

# Function to get volume shape from a NIfTI file
def get_nifti_shape(path):
    try:
        img = nib.load(path)
        return img.shape
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return (None, None, None)

# Apply the shape extraction to each path column
df[['CT_shape_x', 'CT_shape_y', 'CT_shape_z']] = df['CT_Cropped_Path'].apply(lambda x: pd.Series(get_nifti_shape(x)))
df[['PET_original_shape_x', 'PET_original_shape_y', 'PET_original_shape_z']] = df['PET_Original_Cropped_Path'].apply(lambda x: pd.Series(get_nifti_shape(x)))
df[['PET_generated_shape_x', 'PET_generated_shape_y', 'PET_generated_shape_z']] = df['PET_Generated_Cropped_Path'].apply(lambda x: pd.Series(get_nifti_shape(x)))

# Save the updated CSV
df.to_csv('/mimer/NOBACKUP/groups/snic2022-5-277/piacente/my_approach/fold_8_with_shapes.csv', index=False)

# Load CSV
csv_path = '/mimer/NOBACKUP/groups/snic2022-5-277/piacente/my_approach/fold_9.csv'
df = pd.read_csv(csv_path)

# Function to get volume shape from a NIfTI file
def get_nifti_shape(path):
    try:
        img = nib.load(path)
        return img.shape
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return (None, None, None)

# Apply the shape extraction to each path column
df[['CT_shape_x', 'CT_shape_y', 'CT_shape_z']] = df['CT_Cropped_Path'].apply(lambda x: pd.Series(get_nifti_shape(x)))
df[['PET_original_shape_x', 'PET_original_shape_y', 'PET_original_shape_z']] = df['PET_Original_Cropped_Path'].apply(lambda x: pd.Series(get_nifti_shape(x)))
df[['PET_generated_shape_x', 'PET_generated_shape_y', 'PET_generated_shape_z']] = df['PET_Generated_Cropped_Path'].apply(lambda x: pd.Series(get_nifti_shape(x)))

# Save the updated CSV
df.to_csv('/mimer/NOBACKUP/groups/snic2022-5-277/piacente/my_approach/fold_9_with_shapes.csv', index=False)