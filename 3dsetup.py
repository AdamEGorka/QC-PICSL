import SimpleITK as sitk
import gui
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import utilities
matplotlib.use('TkAgg')


def make_isotropic(image, interpolator=sitk.sitkLinear):
    """
    Resample an image to isotropic pixels (using smallest spacing from original) and save to file. Many file formats
    (jpg, png,...) expect the pixels to be isotropic. By default the function uses a linear interpolator. For
    label images one should use the sitkNearestNeighbor interpolator so as not to introduce non-existant labels.
    """
    original_spacing = image.GetSpacing()
    # Image is already isotropic, just return a copy.
    if all(spc == original_spacing[0] for spc in original_spacing):
        return sitk.Image(image)
    # Make image isotropic via resampling.
    original_size = image.GetSize()
    min_spacing = min(original_spacing)
    new_spacing = [min_spacing] * image.GetDimension()
    new_size = [
        int(round(osz * ospc / min_spacing))
        for osz, ospc in zip(original_size, original_spacing)
    ]
    return sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        interpolator,
        image.GetOrigin(),
        new_spacing,
        image.GetDirection(),
        0,
        image.GetPixelID(),
    )


file_path_sr = r"\Users\adame\Desktop\QC Project\2012-02-10_041_S_4509_chunk_sr_left.nii.gz"
file_path_seg = r"\Users\adame\Desktop\QC Project\2012-02-10_041_S_4509_chunk_seg_left.nii.gz"

image_sr = sitk.ReadImage(file_path_sr)
image_sr_data = sitk.GetArrayFromImage(image_sr)

image_seg = sitk.ReadImage(file_path_seg)
image_seg_data = sitk.GetArrayFromImage(image_seg)

image_seg = sitk.Cast(image_seg, sitk.sitkInt32)
print(image_sr.GetSize(), image_seg.GetSize())
# gui.MultiImageDisplay(
#     image_list=[image_sr, image_seg, sitk.LabelToRGB(image_seg)],
#     title_list=["image", "raw segmentation labels", "segmentation labels in color"],
#     figure_size=(9, 3),
#     shared_slider=True,
# )
# Convert the base image to a color image
color_base_image = sitk.Compose([image_sr]*3)  # Create a color image from the grayscale image

# Convert the color images to NumPy arrays
color_base_array = sitk.GetArrayFromImage(color_base_image)
color_array = sitk.GetArrayFromImage(sitk.LabelToRGB(image_seg))

# Make sure both arrays have the same shape
assert color_base_array.shape == color_array.shape, "Arrays must have the same shape"

# Replace the pixel values in the color base array with the corresponding color values
result_array = np.copy(color_base_array)  # Create a copy of the color base array

# Get the indices of non-zero elements in the color array
indices = np.nonzero(color_array)

# Replace the corresponding pixel values in the color base array with color values
result_array[indices] = color_array[indices] * 3.0


# Convert the result array back to a SimpleITK image
result_image = sitk.GetImageFromArray(result_array)

print(result_image.GetSize())
# Display the result image

normalized_array = result_array.astype(np.float32) / np.max(result_array)

slice_result = normalized_array[24, :, :]
plt.imshow(slice_result)

plt.show()
