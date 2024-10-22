import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset

import DataPrepare2
import build_dataset as bd

df = DataPrepare2.get_data_df()


# add oriignal sr
# balances datasets
# vars - acquisition site (too granular?)
# clincal site/site, add gender, age, clinical status to spreadsheet

# Also makes sense to try this first since like Sandy said - the QC ratings are based off of slices

class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()

        # 3D ResNet-18 CNN
        self.cnn = models.video.r3d_18(pretrained=True)

        self.deep_net = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU()
        )
        deep_net_output_size = 4  # Specify the output size of the deep_net module

        # Shared fully connected layer
        cnn_output_size = self.cnn.fc.out_features  # Use the modified output size of the 3D ResNet-18 model
        self.fc = nn.Linear(cnn_output_size + deep_net_output_size, num_classes)

    def forward(self, x1, x2):
        cnn_output = self.cnn(x1)  # Process input with the 3D ResNet-18 CNN
        deep_net_output = self.deep_net(x2)  # Process input with the deep network

        # Concatenate the outputs
        combined_output = torch.cat((cnn_output, deep_net_output), dim=1)
        # Pass through the shared fully connected layer
        output = self.fc(combined_output)

        return output

# Load the pre-trained ResNet-18 model

num_classes = 4
batch_size = 8

model = CustomModel(num_classes)

model_path = r"\Users\adame\PycharmProjects\QCProject\trained_model3d.pt" # Replace with the path to your pre-trained model file
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Define the loss function (e.g., cross-entropy) and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

data_dir = r"C:\Users\adame\Desktop\QC Project\chunkt1_20230512zip\chunkt1_20230512"
#"/data/agorka/QCProject/Data/chunkt1_20230512"
# r"\Users\adame\Desktop\QC Project"
# "C:\Users\adame\Desktop\QC Project\chunkt1_20230512zip\chunkt1_20230512"

# LOAD DATASET #
# Serializing is more convenient because it lets me store the dataset and load it really fast
dataset = bd.load_dataset_from_file(r"\Users\adame\PycharmProjects\QCProject\dataset_serialized")

# If i want to serialize a new dataset
#dataset = bd.new_dataset(data_dir)

#bd.save_dataset_to_file(dataset, "/data/agorka/QCProject/dataset_serialized")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
print("Training size:", len(train_dataset))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Train the model
num_epochs = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):

    # Evaluate the model on the validation set
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    correct_binary = 0

    with torch.no_grad():
        for inputs, labels, age, site, gender, dx1 in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            age_tensor = torch.tensor(age, dtype=torch.float32)
            site_tensor = torch.tensor(site, dtype=torch.long)
            gender_tensor = torch.tensor(gender, dtype=torch.long)
            dx1_tensor = torch.tensor(dx1, dtype=torch.float32)

            input_tensor = torch.cat(
                (age_tensor.unsqueeze(0), site_tensor.unsqueeze(0), gender_tensor.unsqueeze(0), dx1_tensor.unsqueeze(0)),
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

            # Binary correction accuracy - if the predicted and actual are both 0, then success
            #   - If the predicted AND actual are both > 0, then also success (they do not necessarily need the same
            # class rating as long as a problem is identified)
            print(predicted, labels)
            is_both_zero_or_both_non_zero = ((predicted == 0) & (labels == 0)) | (
                    (predicted == 1) & (labels == 1))
            correct_binary += is_both_zero_or_both_non_zero.sum().item()

    # Calculate average validation loss and accuracy for the epoch
    val_loss = val_loss / len(val_loader.dataset)
    val_accuracy = correct / len(val_loader.dataset)
    val_accuracy_binary = correct_binary / len(val_loader.dataset)

    # Print training and validation metrics for the epoch
    print(f"Epoch {epoch + 1}/{num_epochs}:")
    print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f} | Val Binary Accuracy: {val_accuracy_binary:.4f}")

