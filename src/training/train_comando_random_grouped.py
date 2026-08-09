import subprocess

data_path = '/mimer/NOBACKUP/groups/snic2022-5-277/rrestivo/Data/nifti_images'
name = 'RANDOM+GROUPED_district_1'

# Componi il comando da eseguire
# district= ['arms', 'adrenal_gland', 'gallbladder', 'legs', 'kidney', 'spleen', 'thyroid', 'brain', 'liver', 'pancreas', 'stomach', 'lung', 'bladder', 'trachea']
# included_districts=  [arms, adrenal _gland, gallbladder, legs, kidney, spleen, thyroid, brain]
epoch_count= 3476
which_epoch = 'switch10' # --> cambia base_model --> load_network
switch_counter = 11
command = f"python train_nuovo.py --dataroot {data_path} --name {name} --continue_train --gpu_ids 0 --grouped_district --random_district --parto_da_switch --epoch_count {epoch_count}  --which_epoch {which_epoch}"

# Esegui il comando utilizzando subprocess
try:
    subprocess.run(command, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"Errore durante l'esecuzione del comando: {e}")

