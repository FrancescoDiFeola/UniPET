import csv
import os
from monai.data import Dataset, DataLoader, CacheDataset
from monai.transforms import apply_transform
import data.config

class MioDataset(Dataset):
    def __init__(self, opt, folder_paths):
        self.opt = opt
        self.folder_paths = folder_paths
        self.files_A, self.files_B = self._load_files()


    def _load_files(self):
        files_A = []
        files_B = []
        for folder_path in self.folder_paths:
            #print(folder_path)
            main_CT_path = os.path.join(folder_path, 'BOX_CT')
            main_PET_path = os.path.join(folder_path, 'BOX_PET')

            if self.opt.test_district == 'lung':  # ho aggiunto questo
                lung_parts = [
                    ("lung_upper_lobe_right_CT.nii.gz", "lung_upper_lobe_right_PET.nii.gz"),
                    ("lung_middle_lobe_right_CT.nii.gz", "lung_middle_lobe_right_PET.nii.gz"),
                    ("lung_lower_lobe_right_CT.nii.gz", "lung_lower_lobe_right_PET.nii.gz"),
                    ("lung_upper_lobe_left_CT.nii.gz", "lung_upper_lobe_left_PET.nii.gz"),
                    ("lung_lower_lobe_left_CT.nii.gz", "lung_lower_lobe_left_PET.nii.gz"),
                ]
                for ct_file, pet_file in lung_parts:
                    ct_path = os.path.join(main_CT_path, ct_file)
                    pet_path = os.path.join(main_PET_path, pet_file)


                    files_A.append(ct_path)
                    files_B.append(pet_path)


            elif self.opt.test_district == 'kidney':
                kidney_parts = [
                    ("kidney_right_CT.nii.gz", "kidney_right_PET.nii.gz"),
                    ("kidney_left_CT.nii.gz", "kidney_left_PET.nii.gz")
                ]
                for ct_file, pet_file in kidney_parts:
                    ct_path = os.path.join(main_CT_path, ct_file)
                    pet_path = os.path.join(main_PET_path, pet_file)

                    if os.path.exists(ct_path) and os.path.exists(pet_path):
                        files_A.append(ct_path)
                        files_B.append(pet_path)
                    else:
                        print(f"[WARNING] File mancanti per la cartella: {folder_path}")

            elif self.opt.test_district == 'adrenal_gland':
                adrenal_gland_parts = [
                    ("adrenal_gland_right_CT.nii.gz", "adrenal_gland_right_PET.nii.gz"),
                    ("adrenal_gland_left_CT.nii.gz", "adrenal_gland_left_PET.nii.gz")
                ]
                for ct_file, pet_file in adrenal_gland_parts:
                    ct_path = os.path.join(main_CT_path, ct_file)
                    pet_path = os.path.join(main_PET_path, pet_file)

                    if os.path.exists(ct_path) and os.path.exists(pet_path):
                        files_A.append(ct_path)
                        files_B.append(pet_path)
                    else:
                        print(f"[WARNING] File mancanti per la cartella: {folder_path}")

            elif self.opt.test_district == 'thyroid':
                thyroid_parts = [
                    ("thyroid_right_CT.nii.gz", "thyroid_right_PET.nii.gz"),
                    ("thyroid_left_CT.nii.gz", "thyroid_left_PET.nii.gz")
                ]
                for ct_file, pet_file in thyroid_parts:
                    ct_path = os.path.join(main_CT_path, ct_file)
                    pet_path = os.path.join(main_PET_path, pet_file)

                    if os.path.exists(ct_path) and os.path.exists(pet_path):
                        files_A.append(ct_path)
                        files_B.append(pet_path)
                    else:
                        print(f"[WARNING] File mancanti per la cartella: {folder_path}")

            elif self.opt.test_district == 'arms':
                arms_parts = [
                    ("left_arm_CT.nii.gz", "left_arm_PET.nii.gz"),
                    ("right_arm_CT.nii.gz", "right_arm_PET.nii.gz")
                ]
                for ct_file, pet_file in arms_parts:
                    ct_path = os.path.join(main_CT_path, ct_file)
                    pet_path = os.path.join(main_PET_path, pet_file)

                    if os.path.exists(ct_path) and os.path.exists(pet_path):
                        files_A.append(ct_path)
                        files_B.append(pet_path)
                    else:
                        print(f"[WARNING] File mancanti per la cartella: {folder_path}")

            else:
                ct_path = os.path.join(main_CT_path, f'{self.opt.test_district}_CT.nii.gz')
                pet_path = os.path.join(main_PET_path, f'{self.opt.test_district}_PET.nii.gz')
                # Controlla se i file esistono
                if os.path.exists(ct_path) and os.path.exists(pet_path):
                    files_A.append(ct_path)
                    files_B.append(pet_path)
                else:
                    print(f"[WARNING] File mancanti per la cartella: {folder_path}")
                #files_A.append(head_CT_path)
                #files_B.append(head_PET_path)
        return files_A, files_B

    def __getitem__(self, index):
        img_path_A = self.files_A[index]
        img_path_B = self.files_B[index]
        files_dict = [{'A': img_path_A, 'B': img_path_B}]

        # Accedi all'elemento appropriato della lista risultante
        file_dict = files_dict[0]

        return {'A': file_dict['A'], 'B': file_dict['B'], 'A_paths': img_path_A, 'B_paths': img_path_B}

    def __len__(self):
        return len(self.files_B)

def CreateDataloader(opt, shuffle=True, cache=False):
    folder_paths = []

    #with open(f'{opt.dataroot}/{opt.phase}_{opt.district}.csv' if opt.phase=='train' else f'{opt.dataroot}/{opt.phase}.csv', 'r') as csv_file:
    with open( f'{opt.dataroot}/{opt.phase}.csv','r') as csv_file:
        csv_reader = csv.reader(csv_file)
        #next(csv_reader)
        for row in csv_reader:
            folder_paths.append(row[0])

    if not folder_paths:
        print(f"[INFO] Nessun percorso di cartella trovato nel file {opt.phase}_{opt.test_district}.csv")
        return None

    mio_dataset = MioDataset(opt, folder_paths)

    # Decide se utilizzare CacheDataset o Dataset in base al valore di 'cache'
    if cache:
        ds = CacheDataset(data=mio_dataset, transform=data.config.train_transforms if opt.phase=='train' else data.config.test_transforms)
    else:
        ds = Dataset(data=mio_dataset, transform=data.config.train_transforms if opt.phase=='train' else data.config.test_transforms)

    data_loader = DataLoader(ds, batch_size=opt.batchSize, shuffle=shuffle, pin_memory=True)

    return data_loader




