import argparse
import os
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from torch_geometric.data import DataLoader
from torch.nn import BCELoss
from torch.optim import Adam
from dataset_featurizer import MoleculeDataset
from model.GNN1 import GNN1
from model.GNN2 import GNN2
from model.GNN3 import GNN3
import optuna

torch.manual_seed(42)

# Create a parser object
parser = argparse.ArgumentParser(description='GNN Model Training')

# Add arguments for data paths
parser.add_argument('--test_data_path', type=str, required=True, help='Path to the test data file')
parser.add_argument('--train_oversampled', type=str, required=True, help='Path to the train oversampled data file')

# Add an argument for the GNN model selection
parser.add_argument('--model', type=str, choices=['GNN1', 'GNN2', 'GNN3'], default='GNN1', help='Choose the GNN model (GNN1, GNN2, GNN3)')

# Add an argument for the number of epochs
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')

# Parse the command-line arguments
args = parser.parse_args()

# Get the data paths from the command-line arguments
test_data_path = args.test_data_path
train_oversampled_ = args.train_oversampled

# Get the selected model from the command-line arguments
selected_model = args.model

# Get the number of epochs from the command-line arguments
num_epochs = args.epochs

# Load the data
test_data = pd.read_csv(test_data_path)
train_data = pd.read_csv(train_oversampled)

model_folder = "model_weights"
os.makedirs(model_folder, exist_ok=True)

# Define the GNN model based on the selected model
if selected_model == 'GNN1':
    model = GNN1(feature_size=train_data[0].x.shape[1])
  
elif selected_model == 'GNN2':
    model = GNN2(feature_size=train_data[0].x.shape[1])
    
elif selected_model == 'GNN3':
    model = GNN3(feature_size=train_data[0].x.shape[1])
    
else:
    raise ValueError('Invalid model selected')

# Define the loss function
loss_fn = BCELoss()

# Define the optimizer
optimizer = Adam(model.parameters(), lr=0.001)

# Define the objective function for Optuna optimization
def objective(trial):
    # Sample the hyperparameters to be tuned
    hyperparameters = {
        "batch_size": trial.suggest_categorical("batch_size", [32, 128, 64]),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-4, 1e-1),
        "weight_decay": trial.suggest_loguniform("weight_decay", 1e-5, 1e-3),
        "sgd_momentum": trial.suggest_uniform("sgd_momentum", 0.5, 0.9),
        "scheduler_gamma": trial.suggest_categorical("scheduler_gamma", [0.995, 0.9, 0.8, 0.5, 1]),
        "pos_weight": trial.suggest_categorical("pos_weight", [1.0]),
        "model_embedding_size": trial.suggest_categorical("model_embedding_size", [8, 16, 32, 64, 128]),
        "model_attention_heads": trial.suggest_int("model_attention_heads", 1, 4),
        "model_layers": trial.suggest_categorical("model_layers", [3]),
        "model_dropout_rate": trial.suggest_uniform("model_dropout_rate", 0.2, 0.9),
        "model_top_k_ratio": trial.suggest_categorical("model_top_k_ratio", [0.2, 0.5, 0.8, 0.9]),
        "model_top_k_every_n": trial.suggest_categorical("model_top_k_every_n", [0]),
        "model_dense_neurons": trial.suggest_categorical("model_dense_neurons", [16, 128, 64, 256, 32]),
    }

    # Set the hyperparameters in the model
    model.set_hyperparameters(**hyperparameters)

    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=hyperparameters["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=hyperparameters["batch_size"], shuffle=False)

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_fn(out, batch.y)
            loss.backward()
            optimizer.step()

    # Evaluate the model
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch)
            pred = (out >= 0.5).float()
            y_pred.extend(pred.tolist())
            y_true.extend(batch.y.tolist())

    # Compute evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred)

    return f1

# Create the dataset
test_dataset = MoleculeDataset(test_data)
train_dataset = MoleculeDataset(train_data)

# Create the Optuna study and run the optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Get the best hyperparameters and metric
best_hyperparameters = study.best_params
best_metric = study.best_value

# Set the best hyperparameters in the model
model.set_hyperparameters(**best_hyperparameters)

# Create the best data loaders
best_train_loader = DataLoader(train_dataset, batch_size=best_hyperparameters["batch_size"], shuffle=True)
best_test_loader = DataLoader(test_dataset, batch_size=best_hyperparameters["batch_size"], shuffle=False)

# Train the model using the best hyperparameters
for epoch in range(num_epochs):
    model.train()
    for batch in best_train_loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()

    # Evaluate the model
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in best_test_loader:
            out = model(batch)
            pred = (out >= 0.5).float()
            y_pred.extend(pred.tolist())
            y_true.extend(batch.y.tolist())

    # Compute evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred)

    print(f'Epoch {epoch + 1}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, AUC-ROC={auc_roc:.4f}')

print('Best Hyperparameters:', best_hyperparameters)
print('Best Metric:', best_metric)
