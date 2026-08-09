import subprocess

data_path = '/mimer/NOBACKUP/groups/snic2022-5-277/rrestivo/Data/nifti_images'
# Componi il comando da eseguire
which_epoch = 'BEST_final_200'
test_district = 'Whole_body'
name= 'SORTED+GROUPED_district+WARMUP_1'
command = f"python test_patch.py --dataroot {data_path} --gpu_ids 0,1 --name {name} --which_epoch {which_epoch} --test_district {test_district}" # test_stitching
# ricordati di cambiare base_model --> load_network

# Esegui il comando utilizzando subprocess
try:
    subprocess.run(command, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"Errore durante l'esecuzione del comando: {e}")

