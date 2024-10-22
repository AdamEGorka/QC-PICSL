# Ok for now just try transfer learn 2d resnet18 using axial images
import pandas
import DataPrepare
import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import time
from torch.utils.data import Dataset
from PIL import Image
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights
from tempfile import TemporaryDirectory
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

# data
df = DataPrepare.get_data_df()


# Each segmentation is read as a slice. This will make it hard to detect 3d issues but could be useful
# for obvious over-segmentation
# Also makes sense to try this first since like Sandy said - the QC ratings are based off of slices
class MRIDataset(Dataset):
    def __init__(self, data_dir, label_df, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.label_df = label_df

        self.file_list = self.get_file_list()  # Should be a tuple of segs and original sr's

        self.samples = []
        for idx in range(len(self.file_list[0])):
            file_path_sr = self.file_list[0][idx]
            file_path_seg = self.file_list[1][idx]

            image_sr = sitk.ReadImage(file_path_sr)
            image_sr_data = sitk.GetArrayFromImage(image_sr)
            slice_sr = image_sr_data[24, :, :]
            slice_sr = np.fliplr(slice_sr)

            image_seg = sitk.ReadImage(file_path_seg)
            image_seg_data = sitk.GetArrayFromImage(image_seg)
            slice_seg = image_seg_data[24, :, :]

            slice_sr = (slice_sr - np.min(slice_sr)) / (np.max(slice_sr) - np.min(slice_sr))
            slice_seg = (slice_seg - np.min(slice_seg)) / (np.max(slice_seg) - np.min(slice_seg))

            slice_combined = np.hstack((slice_sr, slice_seg))

            label = self.get_label(file_path_seg)

            if label is None:
                continue

            self.samples.append((slice_combined, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

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
        #mtl_value = self.label_df.loc[(self.label_df["ID"] == ID) & (self.label_df["SCANDATE"] == date), mtl_column].values[0]

        mtl_values = self.label_df.loc[
            (self.label_df["ID"] == ID) & (self.label_df["SCANDATE"] == date), mtl_column].values

        if len(mtl_values) > 0:
            mtl_value = mtl_values[0]
        else:
            return None

        label = mtl_value
        print(label)
        return label


# Dataset loading

data_dir = r"\Users\adame\Desktop\QC Project\chunkt1_20230512zip\chunkt1_20230512"  # Just for now
dataset = MRIDataset(data_dir, df)

# Training characteristics
num_samples = len(dataset)
num_epochs = 1
batch_size = 1
print(num_samples)

# plt.imshow(slice, cmap='gray')
# plt.show()
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Set up the model
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# Modify the last fully connected layer for your specific task
num_classes = 3  # Change this according to  number of classes # Pretty sure there are 3 rating types
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Define your loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Apply appropriate transformations to the image (e.g., resizing, normalization)
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Change mean and std if needed
])
# slice = torch.from_numpy(slice).float()
for epoch in range(num_epochs):
    for i, (batch_images, batch_labels) in enumerate(data_loader):
        optimizer.zero_grad()

        # batch_images_transformed = torch.stack([transform(Image.fromarray(img).convert('RGB')).float() for img in
        # batch_images])
        batch_images_transformed = torch.stack(
            [transform(Image.fromarray(img.numpy()).convert('RGB')).float() for img in batch_images])

        # batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.long)
        batch_labels_tensor = batch_labels.clone().detach().long()

        outputs = model(batch_images_transformed)

        loss = criterion(outputs, batch_labels_tensor)

        loss.backward()

        optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(data_loader)}], Loss: {loss.item()}")



# Now make it into a loop
# slice_pil = Image.fromarray(slice)
# slice_pil = slice_pil.convert('RGB')

# image = transform(slice_pil)

# print(image.shape)

# Forward pass
# outputs = model(image.unsqueeze(0))
# loss = criterion(outputs, label)

# Backward pass and optimization
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()
# print(loss.item())
