import pickle
import random

import torch
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import zoom
from datetime import datetime
import os
from torch.utils.data import Dataset

import DataPrepare2


def save_dataset_to_file(dataset_instance, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dataset_instance, file)


def load_dataset_from_file(filename):
    with open(filename, 'rb') as file:
        dataset_instance = pickle.load(file)
    return dataset_instance


def new_dataset(data_dir, transform=None):
    df = DataPrepare2.get_data_df()
    dataset = MRIDataset(data_dir, df, transform)
    return dataset


class MRIDataset(Dataset):
    def __init__(self, data_dir, label_df, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.label_df = label_df

        self.file_list = self.get_file_list()

        self.samples = []
        num_items = len(self.file_list[0])

        for idx in range(num_items):  # change range if need a temporary limit
            file_path_sr = self.file_list[0][idx]
            file_path_seg = self.file_list[1][idx]

            label = self.get_label(file_path_seg)
            if label is None:
                print("Error finding label")
                continue

            image_sr = sitk.ReadImage(file_path_sr)
            image_sr_data = sitk.GetArrayFromImage(image_sr)

            image_seg = sitk.ReadImage(file_path_seg)
            image_seg = sitk.Cast(image_seg, sitk.sitkInt32)
            image_seg_data = sitk.GetArrayFromImage(image_seg)

            target_size = (43, 124, 124)
            scale_factors_sr = tuple(target_size[i] / image_sr_data.shape[i] for i in range(3))
            scale_factors_seg = tuple(target_size[i] / image_seg_data.shape[i] for i in range(3))

            image_sr_data = zoom(image_sr_data, scale_factors_sr)
            image_seg_data = zoom(image_seg_data, scale_factors_seg)

            # Rescaling in order to do composites
            image_sr = sitk.GetImageFromArray(image_sr_data)
            image_seg = sitk.GetImageFromArray(image_seg_data)
            image_seg = sitk.Cast(image_seg, sitk.sitkFloat32)

            rescaler = sitk.RescaleIntensityImageFilter()
            rescaler.SetOutputMaximum(255)
            rescaler.SetOutputMinimum(0)
            image_sr = rescaler.Execute(image_sr)
            image_seg = rescaler.Execute(image_seg)
            difference_vol = image_sr - image_seg

            # Create a 3 channel image where first two channels are original images and 3rd is a difference volume
            composite = sitk.Compose(image_sr, image_seg, difference_vol)

            composite_array = sitk.GetArrayFromImage(composite)  # Get it as a numpy array
            # Apply random rotations and flips to prevent overtraining on a specific orientation
            random_rotations = random.randint(0,3)
            random_flips = random.randint(0,1)
            composite_array = np.rot90(composite_array, random_rotations, axes=(1,2))
            if random_flips == 1:
                composite_array = np.flip(composite_array, axis=1)

            print(composite_array.shape)

            # Also need to add all other variables here
            file_name = os.path.basename(file_path_seg)
            date = file_name.split("_")[0]
            ID = "_".join(file_name.split("_")[1:-3])

            date_str = date
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            formatted_date_str = '{}/{}'.format(date_obj.month, date_obj.day) + '/' + str(date_obj.year)[-2:]

            filtered_df = self.label_df.loc[
                (self.label_df["ID"] == ID) & (self.label_df["SCANDATE"] == formatted_date_str)]

            age = filtered_df["AGE"].values[0]
            site = filtered_df["SITE"].values[0]
            gender = filtered_df["PTGENDER"].values[0]
            gender_binary = 0 if gender == "Male" else 1
            dx1 = filtered_df["DX1"].values[0]

            self.samples.append((composite_array, label, age, site, gender_binary, dx1))

    def resize_3d_image(self, image, new_shape):
        # Calculate the resize factors for each dimension
        resize_factors = np.array(new_shape) / np.array(image.shape)

        # Resize the image using the zoom function
        resized_image = zoom(image, resize_factors, mode='nearest')

        return resized_image

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        result_array, label, age, site, gender_binary, dx1 = self.samples[idx]
        result_array = torch.from_numpy(result_array).float()
        result_array = result_array.permute(3, 0, 1, 2)
        # Reorder dimensions to have shape [channels, height, width, depth]

        label = torch.tensor(float(label)).long()
        age = torch.tensor(float(age)).long()
        site = torch.tensor(float(site)).long()
        gender_binary = torch.tensor(float(gender_binary)).long()
        dx1 = torch.tensor(float(dx1)).long()

        return result_array, label, age, site, gender_binary, dx1

    def get_file_list(self):
        file_list_seg = [os.path.join(self.data_dir, file) for file in os.listdir(self.data_dir) if
                         file.endswith('.gz') and "seg" in file]
        file_list_sr = [os.path.join(self.data_dir, file) for file in os.listdir(self.data_dir) if
                        file.endswith('.gz') and "sr" in file]
        return file_list_sr, file_list_seg

    def get_label(self, file_path):
        file_name = os.path.basename(file_path)
        date = file_name.split("_")[0]
        ID = "_".join(file_name.split("_")[1:-3])
        print(date, ID)

        mtl_column = "MTL_LEFT" if "left" in file_name.lower() else "MTL_RIGHT"

        date_str = date
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        formatted_date_str = '{}/{}'.format(date_obj.month, date_obj.day) + '/' + str(date_obj.year)[-2:]

        mtl_values = self.label_df.loc[
            (self.label_df["ID"] == ID) & (self.label_df["SCANDATE"] == formatted_date_str), mtl_column].values

        if len(mtl_values) > 0:
            mtl_value = mtl_values[0]
            print(mtl_values[0])
        else:
            return None

        label = mtl_value
        return label
