import random
import csv
import os
from monai.data import Dataset, DataLoader, CacheDataset
from monai.transforms import apply_transform
import data.config

class MioDataset(Dataset):
    def __init__(self, opt, folder_paths_per_district, included_districts):
        self.opt = opt
        self.folder_paths_per_district = folder_paths_per_district
        self.all_files_A = []
        self.all_files_B = []
        self.all_labels = []
        self.included_districts = included_districts

        # Carica i file per ciascun distretto
        for district_index, folder_paths in enumerate(folder_paths_per_district):
            district_files_A, district_files_B, district_labels = self._load_files(folder_paths, district_index)
            self.all_files_A.append(district_files_A)
            self.all_files_B.append(district_files_B)
            self.all_labels.append(district_labels)  

        self.img_transform = data.config.train_transforms

    def _load_files(self, folder_paths, district_index):
        files_A = []
        files_B = []
        labels = []

        district = self.included_districts[district_index]

        for folder_path in folder_paths:
            main_CT_path = os.path.join(folder_path, 'BOX_CT')
            main_PET_path = os.path.join(folder_path, 'BOX_PET')
            #print(f'distretto: {self.opt.district[district_index]}')
            if district == 'lung': # ho aggiunto questo
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
                    labels.append(district_index) # Aggiungi la stessa etichetta per tutte le parti

            elif district == 'kidney':
                kidney_parts = [
                    ("kidney_right_CT.nii.gz", "kidney_right_PET.nii.gz"),
                    ("kidney_left_CT.nii.gz", "kidney_left_PET.nii.gz")
                ]
                for ct_file, pet_file in kidney_parts:
                    ct_path = os.path.join(main_CT_path, ct_file)
                    pet_path = os.path.join(main_PET_path, pet_file)

                    files_A.append(ct_path)
                    files_B.append(pet_path)
                    labels.append(district_index)

            elif district == 'adrenal_gland':
                adrenal_gland_parts = [
                    ("adrenal_gland_right_CT.nii.gz", "adrenal_gland_right_PET.nii.gz"),
                    ("adrenal_gland_left_CT.nii.gz", "adrenal_gland_left_PET.nii.gz")
                ]
                for ct_file, pet_file in adrenal_gland_parts:
                    ct_path = os.path.join(main_CT_path, ct_file)
                    pet_path = os.path.join(main_PET_path, pet_file)

                    files_A.append(ct_path)
                    files_B.append(pet_path)
                    labels.append(district_index)

            elif district == 'thyroid':
                thyroid_parts = [
                    ("thyroid_right_CT.nii.gz", "thyroid_right_PET.nii.gz"),
                    ("thyroid_left_CT.nii.gz", "thyroid_left_PET.nii.gz")
                ]
                for ct_file, pet_file in thyroid_parts:
                    ct_path = os.path.join(main_CT_path, ct_file)
                    pet_path = os.path.join(main_PET_path, pet_file)

                    files_A.append(ct_path)
                    files_B.append(pet_path)
                    labels.append(district_index)

            elif district == 'arms':
                arms_parts = [
                    ("left_arm_CT.nii.gz", "left_arm_PET.nii.gz"),
                    ("right_arm_CT.nii.gz", "right_arm_PET.nii.gz")
                ]
                for ct_file, pet_file in arms_parts:
                    ct_path = os.path.join(main_CT_path, ct_file)
                    pet_path = os.path.join(main_PET_path, pet_file)

                    files_A.append(ct_path)
                    files_B.append(pet_path)
                    labels.append(district_index)

            else:
                district_CT_path = os.path.join(main_CT_path, f'{self.opt.district[district_index]}_CT.nii.gz')
                district_PET_path = os.path.join(main_PET_path, f'{self.opt.district[district_index]}_PET.nii.gz')

                files_A.append(district_CT_path)
                files_B.append(district_PET_path)
                labels.append(district_index)

        return files_A, files_B, labels

    def __getitem__(self, index):
        # Seleziona un distretto casualmente
        district_index = random.randint(0, len(self.all_files_A) - 1)

        # Seleziona l'indice per il distretto scelto
        district_files_A = self.all_files_A[district_index]
        district_files_B = self.all_files_B[district_index]
        district_labels = self.all_labels[district_index]

        # Usa modulo per gestire gli indici fuori dai limiti
        index_in_district = index % len(district_files_B)

        img_path_A = district_files_A[index_in_district]
        img_path_B = district_files_B[index_in_district]
        label = district_labels[index_in_district]

        return {
            'A': img_path_A, 'B': img_path_B,
            'A_paths': img_path_A, 'B_paths': img_path_B,
            'label': label  # Restituisci l'etichetta per il distretto
        }

    def __len__(self):
        # La lunghezza del dataset è basata sul distretto più grande
        return max(len(files_B) for files_B in self.all_files_B)


def CreateDynamicDataloader(opt, included_districts, shuffle=True, cache=False):
    folder_paths_per_district = []

    # Carica i percorsi per ciascun distretto incluso
    for district_name in included_districts:
        folder_paths = []
        csv_path = f'{opt.dataroot}/{opt.phase}_{district_name}.csv'
        with open(csv_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                folder_paths.append(row[0])
        folder_paths_per_district.append(folder_paths)

    # Crea il dataset dinamico
    mio_dataset = MioDataset(opt, folder_paths_per_district, included_districts)

    # Usa CacheDataset o Dataset in base al flag `cache`
    if cache:
        ds = CacheDataset(data=mio_dataset, transform=data.config.train_transforms if opt.phase == 'train' else data.config2.test_transforms, cache_rate=0.10)
    else:
        ds = Dataset(data=mio_dataset, transform=data.config.train_transforms if opt.phase == 'train' else data.config2.test_transforms)

    # Crea il dataloader
    data_loader = DataLoader(ds, batch_size=opt.batchSize, shuffle=shuffle, pin_memory=True)

    return data_loader
