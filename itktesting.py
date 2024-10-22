import SimpleITK as sitk
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.ndimage import rotate
import torch
import random

import cv2

def apply_random_rotation(image, random_rot, random_flips):
    image = rotate(image, random_rot, axes=(0, 1), reshape=False)
    if random_flips == 1:
        image = np.flip(image, axis=1)
    return image

# Channel 1 - sr images
# Channel 2 - seg images (I think this will be okay
# Channel 3 - make it 0 or adjust model to not need a 3rd channel

# Load the original MRI image and the segmentation
image_seg = sitk.ReadImage(r"\Users\adame\Desktop\QC Project\2012-05-17_072_S_4522_chunk_seg_right.nii.gz")
image_sr = sitk.ReadImage(r"C:\Users\adame\Desktop\QC Project\2012-05-17_072_S_4522_chunk_sr_right.nii.gz")

# Convert the segmentation to a binary image

image_seg_data = sitk.GetArrayFromImage(image_seg)
image_sr_data = sitk.GetArrayFromImage(image_sr)
# 20, 40 ,60 ,80 100 for coronal slices
# 16.6% increments all the way up to 88% coronal slice

# now i need to split up the sr slices (keep them on top tho)

# Resize both images to same dimensions (guarantee they are the same size)
print(image_sr_data.shape)
# Do i actually need to resize though? - might need to change this later
target_size = (100, 100, 100)
scale_factors_sr = tuple(target_size[i] / image_sr_data.shape[i] for i in range(3))
image_sr_data = zoom(image_sr_data, scale_factors_sr)  # resize
scale_factors_seg = tuple(target_size[i] / image_seg_data.shape[i] for i in range(3))
image_seg_data = zoom(image_seg_data, scale_factors_seg)  # resize

# CORONAL 33% to 84
depth = image_sr_data.shape[0]  # perhaps readjust size of both images before doing this ?
start = .166
end = .88
increment = .166
start_index = int(depth * start)
end_index = int(depth * end)

sr_row = image_sr_data[start_index, :, :]
seg_row = image_seg_data[start_index, :, :]

random_rot = random.randint(0, 360)
random_flips = random.randint(0, 1)

sr_row = apply_random_rotation(sr_row, random_rot, random_flips)
seg_row = apply_random_rotation(seg_row, random_rot, random_flips)

print(random_rot, random_flips)

for index in range(start_index + int(depth * increment), end_index + 1, int(depth * increment)):
    image_sr_slice = image_sr_data[index, :, :]
    image_seg_slice = image_seg_data[index, :, :]
    image_sr_slice = apply_random_rotation(image_sr_slice, random_rot, random_flips)
    image_seg_slice = apply_random_rotation(image_seg_slice, random_rot, random_flips)
    sr_row = np.concatenate((sr_row, image_sr_slice), axis=1)
    seg_row = np.concatenate((seg_row, image_seg_slice), axis=1)

# SAGITTAL
depth = image_sr_data.shape[2]
start = .33
end = .84
increment = .33
start_index = int(depth * start)
end_index = int(depth * end)

for index in range(start_index, end_index + 1, int(depth * increment)):
    print(index)
    image_seg_slice = image_seg_data[:, :, index]
    image_sr_slice = image_sr_data[:, :, index]
    image_sr_slice = apply_random_rotation(image_sr_slice, random_rot, random_flips)
    image_seg_slice = apply_random_rotation(image_seg_slice, random_rot, random_flips)
    sr_row = np.concatenate((sr_row, image_sr_slice), axis=1)
    seg_row = np.concatenate((seg_row, image_seg_slice), axis=1)

# mean_value = np.mean(concat_resized)
# std_value = np.std(concat_resized)
# concat_normalized = (concat_resized - mean_value) / std_value


print(sr_row.shape, seg_row.shape)

# Normalize SR image
mean_sr = np.mean(sr_row)
std_sr = np.std(sr_row)
sr_row_normalized = (sr_row - mean_sr) / std_sr

# Normalize Seg image
# For now im just leaving it as is
sr_row = np.expand_dims(sr_row_normalized, axis=0)  # Expand dimensions to (height, width, 1)
seg_row = np.expand_dims(seg_row, axis=0)  # Expand dimensions to (height, width, 1)
zeros_channel = np.zeros_like(sr_row)  # Create an all-zero channel

# Stack the images along the third axis to create the 3D image
combined_image = np.concatenate((sr_row, seg_row, zeros_channel), axis=0)

print(combined_image.shape)

plt.subplot(1, 2, 1)
plt.imshow(combined_image[0], cmap='gray')
plt.title('SR')

plt.subplot(1, 2, 2)
plt.imshow(combined_image[1], cmap='gray')  # Change 'another_image' to your actual image data
plt.title('SEG')  # Set a title for the second subplot

plt.tight_layout()
plt.show()
# plt.imshow(overlay_slice)
# plt.title('MRI Image with Segmentation Overlay')
# plt.colorbar()
# plt.show()

# Display the overlay
