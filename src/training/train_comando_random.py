import subprocess

data_path = '/mimer/NOBACKUP/groups/snic2022-5-277/rrestivo/Data/nifti_images'
name = 'RANDOM_district'
epoch_count= 5099
which_epoch = 'switch13' # --> cambia base_model --> load_network
switch_counter = 14
# Componi il comando da eseguire

command = f"python train_nuovo.py --dataroot {data_path} --name {name} --continue_train --gpu_ids 0 --random_district --parto_da_switch --which_epoch {which_epoch} --switch_counter {switch_counter} --epoch_count {epoch_count}"

# Esegui il comando utilizzando subprocess
try:
    subprocess.run(command, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"Errore durante l'esecuzione del comando: {e}")

