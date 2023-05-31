import torch
import pandas as pd
from dataset_featurizer import MoleculeDataset
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

# Load the test dataset
test_dataset = MoleculeDataset(root="data/split_data", filename="HIV_test.csv", test=True)
test_loader = DataLoader(test_dataset, batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)

# Load the trained model
model = torch.load(os.join.path(output_folder,"model.pth"))
model.eval()

# Create lists to store the predicted and true labels
all_preds = []
all_labels = []
all_preds_raw = []

# Perform inference on the test dataset
with torch.no_grad():
    for batch in test_loader:
        # Move the batch to the device
        batch = batch.to(device)

        # Perform forward pass
        pred = model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.batch)

        # Convert the predictions to class labels
        preds = torch.argmax(pred, dim=1)

        # Append the predicted and true labels to the lists
        all_preds.extend(preds.cpu().detach().numpy())
        all_labels.extend(batch.y.cpu().detach().numpy())
        all_preds_raw.extend(pred.cpu().detach().numpy())

# Calculate the confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Calculate the accuracy score
accuracy = accuracy_score(all_labels, all_preds)

# Calculate the ROC AUC score
roc_auc = roc_auc_score(all_labels, all_preds_raw)

# Print the confusion matrix, accuracy score, and ROC AUC score
print("Confusion Matrix:")
print(cm)
print("Accuracy Score:", accuracy)
print("ROC AUC Score:", roc_auc)