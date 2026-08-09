import subprocess

data_path = '/mimer/NOBACKUP/groups/snic2022-5-277/rrestivo/Data/nifti_images'
name = 'SORTED+GROUPED_district_1'

# Componi il comando da eseguire
epoch_count= 4004
which_epoch = 'switch10' # --> cambia base_model --> load_network
switch_counter = 11
#included_districts= ['gallbladder', 'thyroid','bladder', 'trachea', 'kidney', 'spleen', 'pancreas', 'stomach'] --> train options

command = f"python train_nuovo.py --dataroot {data_path} --name {name} --continue_train --gpu_ids 0 --grouped_district --parto_da_switch --epoch_count {epoch_count} --switch_counter {switch_counter} --which_epoch {which_epoch} "

# Esegui il comando utilizzando subprocess
try:
    subprocess.run(command, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"Errore durante l'esecuzione del comando: {e}")

