###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data
from PIL import Image
import os
import os.path
import csv
import nibabel as nib  # Aggiunto per leggere immagini NIfTI (.nii.gz)

IMG_EXTENSIONS = [
    '.nii.gz',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset_from_csv(csv_path):
    images = []
    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Salta l'intestazione se presente
        for row in csv_reader:
            folder_path = row[0]  # Assumi che la colonna contenga i percorsi delle cartelle
            head_CT_path = os.path.join(folder_path, 'BOX_CT', 'head_CT.nii.gz')
            head_PET_path = os.path.join(folder_path, 'BOX_PET', 'head_PET.nii.gz')
            images.append((head_CT_path, head_PET_path))
    return images

def default_loader_nifti(path):
    img = nib.load(path).get_fdata()  # Carica l'immagine NIfTI
    return img

class NiftiDataset(data.Dataset):

    def __init__(self, csv_path, transform=None, return_paths=False,
                 loader=default_loader_nifti):
        imgs = make_dataset_from_csv(csv_path)
        if len(imgs) == 0:
            raise RuntimeError(f"Found 0 images in: {csv_path}\n"
                               f"Supported image extensions are: {','.join(IMG_EXTENSIONS)}")

        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        ct_path, pet_path = self.imgs[index]
        ct_img = self.loader(ct_path)
        pet_img = self.loader(pet_path)

        if self.transform is not None:
            ct_img = self.transform(ct_img)
            pet_img = self.transform(pet_img)

        if self.return_paths:
            return ct_img, pet_img, ct_path, pet_path
        else:
            return ct_img, pet_img

    def __len__(self):
        return len(self.imgs)

