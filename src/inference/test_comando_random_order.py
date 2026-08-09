import subprocess

data_path = '/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/ct2pet/ENHANCE_PET_MOOSE_1_6k'  # '/mimer/NOBACKUP/groups/snic2022-5-277/rrestivo/Data/nifti_images' 
# Componi il comando da eseguire
name = 'RANDOM_district_1'
which_epoch = 'best'

test_district = 'liver'

command = f"python test_stitching_clean.py --dataroot {data_path} --which_epoch {which_epoch} --name {name}  --test_district {test_district}" # test_stitching


# Esegui il comando utilizzando subprocess
try:
    subprocess.run(command, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"Errore durante l'esecuzione del comando: {e}")

