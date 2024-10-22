import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset
import numpy as np

import DataPrepare2
import build_dataset2d as bd
import torch.nn.functional as F

df = DataPrepare2.get_data_df()


# add oriignal sr
# balances datasets
# vars - acquisition site (too granular?)
# clincal site/site, add gender, age, clinical status to spreadsheet, and IMAGE QUALITY!

# Also makes sense to try this first since like Sandy said - the QC ratings are based off of slices
# Try without sagittal slices

# SEt downsampling paramters

# meeting notes
# Add other variables to the dataset (like patient status)
#
#         Accuracy  Pos Acc.
# Resnet18 - 87%    80.65%
# Resnet50 - 86%    81.05%
# Resnet101 - 88.29%   80.16%

# LAter goals: (kinda for report)
# Cross validation sets
# See if classification works better in certain categories of subjects

# Note - dx1 sometimes has values of -9.2234e+18, - its reading null rows as -9.2234e+18 (drop them or replace with a 0 maybe)

# maybe try adding the scalars through the resnet model by connecting it into resnet's first fully connected layer
# also adni already provides image quality (INCLUDE IT)
# train resnet first, then freeze convolutional paramters, and only add scalars to fully connected layer
# rohan, adni quality, site, clinical status

# OR
# during the learning process let it learn the image quality (set the output to 2 numbers) 2 loss functions and multiclass learning
# set quality as auxilliary task
# check graphs to see if other values/scalars are correlated with qc values

# either add multiclass auxillary loss OR add scalars to the resnet model

class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()

        # 2d mode no extra scalars
        # self.resnet = models.resnet18(pretrained=False)
        # #self.resnet.conv1 = nn.Conv2d(43, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # # Shared fully connected layer
        # cnn_output_size = self.resnet.fc.out_features  # Use the modified output size of the 3D ResNet-18 model
        # self.resnet.fc = nn.Linear(512, num_classes)

        # ~~~~~~~~~ 2d mode with extra scalars ~~~~~~~~~
        # resnet
        self.resnet = models.resnet101(pretrained=True)
        cnn_output_size = self.resnet.fc.out_features  # Use the modified output size of the 2D model

        # parallel network
        self.deep_net = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU()
        )
        deep_net_output_size = 4
        self.fc = nn.Linear(cnn_output_size + deep_net_output_size, num_classes)

    def forward(self, x1, x2):
        # ~~~ 2d node no extra scalars ~~~
        # output = self.resnet(x1)

        # ~~~ 2d mode with extra scalars ~~~
        # print('resnet architechture: ', self.resnet)
        # print('cnn input shape: ', x1.shape)
        cnn_output = self.resnet(x1)
        dn_output = self.deep_net(x2)
        combined_out = torch.cat((cnn_output, dn_output), dim=1)
        output = self.fc(combined_out)

        return output


# Load the pre-trained ResNet-18 model

num_classes = 2
batch_size = 64

model = CustomModel(num_classes)

model_path = "/data/agorka/QCProject/trained_model_50_2d_8_16_slices_binary.pt"  # Replace with the path to your pre-trained model file
# model.load_state_dict(torch.load(model_path))

# Define the loss function (e.g., cross-entropy) and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

data_dir = "/data/agorka/QCProject/Data/chunkt1_20230512"
# r"\Users\adame\Desktop\QC Project"
# "C:\Users\adame\Desktop\QC Project\chunkt1_20230512zip\chunkt1_20230512"

# LOAD DATASET #
# Serializing is more convenient because it lets me store the dataset and load it really fast
dataset = bd.load_dataset_from_file("/data/agorka/QCProject/dataset_serialized2d_slices_binary")

# If i want to serialize a new dataset
# dataset = bd.new_dataset(data_dir)
# bd.save_dataset_to_file(dataset, "/data/agorka/QCProject/dataset_serialized2d_slices_binary")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
# Get even splitting of the training and validation datasets
label_indices = [[] for _ in range(3)]  # 3 represents the number of label types (0, 1, and 2)

# Step 2: Loop through the dataset and collect the indices for each label type
for idx, (_, label, _, _, _, _) in enumerate(dataset):
    # if label in [1, 2]:  # Only consider labels 1 and 2
    # label = int(label)
    label_indices[label].append(idx)

class_counts = [len(label_indices[class_idx]) for class_idx in range(num_classes)]
total_samples = sum(class_counts)
class_weights = [total_samples / (num_classes * count) for count in class_counts]
class_weights = torch.tensor(class_weights, dtype=torch.float)
print("Class counts", class_counts, class_weights)

class_weights[1] = class_weights[1] * 1.2
criterion = nn.CrossEntropyLoss(weight=class_weights)  # seems to not be working as well

min_samples_per_class = min(class_counts)
train_proportions = [int(0.8 * class_counts[class_idx]) for class_idx in range(num_classes)]

print("Training proportions: ", train_proportions)

# num sampless to take for each label type
# train_proportions = [int(0.8 * len(indices)) for indices in label_indices]
print(train_proportions)
train_indices = []
val_indices = []

for class_idx in range(num_classes):
    indices = label_indices[class_idx]

    np.random.seed(42)  # For reproducibility
    shuffled_indices = np.random.permutation(indices)

    # Append the first 'train_proportions[class_idx]' indices to the training set and the rest to the validation set
    train_indices.extend(shuffled_indices[:train_proportions[class_idx]])
    val_indices.extend(shuffled_indices[train_proportions[class_idx]:])
    print(class_idx, len(train_indices), len(val_indices))

    train_size = train_proportions[class_idx]
    val_size = len(indices) - train_size

# Use the train_indices and val_indices to create the final datasets (this dataset will include randomly proortional amounts of each class)
train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)
print("val size", len(val_dataset))
print("Training size:", len(train_dataset))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Train the model
num_epochs = 50
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
class_weights = class_weights.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    train_loss = 0.0

    for inputs, labels, age, site, gender, dx1 in train_loader:
        age_tensor = torch.tensor(age, dtype=torch.float32)
        site_tensor = torch.tensor(site, dtype=torch.long)
        gender_tensor = torch.tensor(gender, dtype=torch.long)
        dx1_tensor = torch.tensor(dx1, dtype=torch.float32)

        input_tensor = torch.cat(
            (age_tensor.unsqueeze(0), site_tensor.unsqueeze(0), gender_tensor.unsqueeze(0), dx1_tensor.unsqueeze(0)),
            dim=0)
        input_tensor = input_tensor.permute(1, 0)
        # input_tensor = torch.cat((age_tensor, site_tensor, gender_tensor, dx1_tensor), dim=0)
        # print('age', age_tensor.shape, age_tensor)
        # print('site', site_tensor.shape, site_tensor)
        # print('gender', gender_tensor.shape, gender_tensor)
        # print('dx1', dx1_tensor.shape, dx1_tensor)
        # print('input tensor', input_tensor.shape, input_tensor)

        inputs = inputs.to(device)
        labels = labels.to(device)
        input_tensor = input_tensor.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs, input_tensor)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

    # Calculate average training loss for the epoch
    train_loss = train_loss / len(train_loader.dataset)
    print(train_loss)

    # Evaluate the model on the validation set
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    binary_correct = 0
    binary_total = 0

    with torch.no_grad():
        for inputs, labels, age, site, gender, dx1 in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            age_tensor = torch.tensor(age, dtype=torch.float32)
            site_tensor = torch.tensor(site, dtype=torch.long)
            gender_tensor = torch.tensor(gender, dtype=torch.long)
            dx1_tensor = torch.tensor(dx1, dtype=torch.float32)

            input_tensor = torch.cat(
                (
                    age_tensor.unsqueeze(0), site_tensor.unsqueeze(0), gender_tensor.unsqueeze(0),
                    dx1_tensor.unsqueeze(0)),
                dim=0)
            input_tensor = input_tensor.permute(1, 0)
            input_tensor = input_tensor.to(device)

            # Forward pass and calculate loss
            outputs = model(inputs, input_tensor)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

            # Calculate the number of correctly classified samples
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

            # Binary accuracy
            binary_correct += ((predicted == labels) & (labels == 1)).sum().item()
            binary_total += (labels == 1).sum().item()
            label_counts = torch.bincount(labels)
            print(predicted, labels, label_counts)

    # Calculate average validation loss and accuracy for the epoch
    val_loss = val_loss / len(val_loader.dataset)
    val_accuracy = correct / len(val_loader.dataset)

    binary_accuracy = binary_correct / binary_total
    print(binary_correct, binary_total)
    # Print training and validation metrics for the epoch
    print(f"Epoch {epoch + 1}/{num_epochs}:")
    print(
        f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f} | Label 1 Accuracy: {binary_accuracy:.4f}")
    if epoch == num_epochs - 1:
        print(predicted, labels)

torch.save(model.state_dict(), "/data/agorka/QCProject/trained_model_18_2d_8_23_slices_scalars_binary.pt")
