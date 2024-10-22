import torch
import torch.nn as nn
import torchvision.models as models


# Custom model class
class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()

        # 3D ResNet-18 CNN
        self.cnn = models.video.r3d_18(pretrained=True)
        self.cnn.fc = nn.Identity()  # Remove the original fully connected layer

        # Parallel deep network
        self.deep_net = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU()
        )

        # Shared fully connected layer
        self.fc = nn.Linear(4 + 512, num_classes)  # 4 for deep_net output, 512 for cnn output

    def forward(self, x1, x2):
        cnn_output = self.cnn(x1)  # Process input with the 3D ResNet-18 CNN
        deep_net_output = self.deep_net(x2)  # Process input with the deep network

        # Concatenate the outputs
        combined_output = torch.cat((cnn_output, deep_net_output), dim=1)

        # Pass through the shared fully connected layer
        output = self.fc(combined_output)

        return output


# Instantiate the model
num_classes = 10  # Change this according to your task
model = CustomModel(num_classes)

# Test the model with random input tensors
x1 = torch.randn(1, 3, 43, 124, 124)  # Example input for the 3D ResNet-18 CNN
x2 = torch.randn(1, 4)  # Example input for the deep network
output = model(x1, x2)
print(output.shape)  # Should print: torch.Size([1, num_classes])
