import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset

import DataPrepare2
import build_dataset as bd


true_labels_list = []
predicted_probs_list = []

model.eval()  # Set the model to evaluation mode

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

        # Forward pass and calculate predicted probabilities
        outputs = model(inputs, input_tensor)
        softmax_outputs = nn.Softmax(dim=1)(outputs)
        predicted_probs = softmax_outputs.cpu().numpy()

        # Convert labels to numpy array
        true_labels_np = labels.cpu().numpy()

        # Append true labels and predicted probabilities to the lists
        true_labels_list.extend(true_labels_np)
        predicted_probs_list.extend(predicted_probs)

# Convert the lists to numpy arrays
true_labels_np = np.array(true_labels_list)
predicted_probs_np = np.array(predicted_probs_list)

# Compute the ROC-AUC score
roc_auc = roc_auc_score(true_labels_np, predicted_probs_np, multi_class='ovr')

print("ROC-AUC Score:", roc_auc)