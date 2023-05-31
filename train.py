#%%
import os
import rdkit
import torch
import cairosvg
import numpy as np
import torch_geometric
import pandas as pd
from tqdm import tqdm
import deepchem as dc
from PIL import Image
from rdkit import Chem
from rdkit import Chem
from rdkit import RDConfig
from rdkit.Chem import Draw
from rdkit.Chem import Draw
from rdkit.Chem import rdBase
import matplotlib.pyplot as plt
import IPython.display as display
from rdkit.Chem.Draw import IPythonConsole
from sklearn.model_selection import train_test_split
from torch_geometric.data import Dataset, Data
from torch_geometric.data import DataLoader
from dataset_featurizer import MoleculeDataset
from gnn_project.model.GNN1 import GNN1
from gnn_project.model.GNN2 import GNN2
from gnn_project.model.GNN3 import GNN3
import torch.nn.functional as F 
import seaborn as sns
from torch.nn import Sequential, Linear, BatchNorm1d, ModuleList, ReLU
from torch_geometric.nn import TransformerConv, TopKPooling, GATConv, BatchNorm
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from sklearn.metrics import confusion_matrix, f1_score, \
    accuracy_score, precision_score, recall_score, roc_auc_score
from torch_geometric.nn.conv.x_conv import XConv
torch.manual_seed(42)

data_path = 'data/raw/HIV_data.csv'
data = pd.read_csv(data_path,index_col=[0])
print("###### Raw Data Shape - ")
print(data.shape)
print("0 : ",data["HIV_active"].value_counts()[0])
print("1 : ",data["HIV_active"].value_counts()[1])

#%%
# Split the data, Due to imbalance datasets we need to set train ration high
print("###### After Data split Shape - ")
train_data = pd.read_csv('data/split_data/HIV_train.csv')
test_data = pd.read_csv('data/split_data/HIV_test.csv')
train_data_oversampled = pd.read_csv('data/split_data/HIV_train_oversampled.csv')
print("Train Data ", train_data.shape)
print("0 : ",train_data["HIV_active"].value_counts()[0])
print("1 : ",train_data["HIV_active"].value_counts()[1])

print("Test Data ", test_data.shape)
print("0 : ",test_data["HIV_active"].value_counts()[0])
print("1 : ",test_data["HIV_active"].value_counts()[1])
 
print("Train Data Oversampled ", train_data_oversampled.shape)
print("0 : ",train_data_oversampled["HIV_active"].value_counts()[0])
print("1 : ",train_data_oversampled["HIV_active"].value_counts()[1])


# %%
# Define the folder path to save the images
output_folder = "visualization"
os.makedirs(output_folder, exist_ok=True)

sample_smiles = train_data["smiles"][4:30].values
sdf = Chem.SDMolSupplier(output_folder+'/cdk2.sdf')
mols = [m for m in sdf]

for i, smiles in enumerate(sample_smiles):
    core = Chem.MolFromSmiles(smiles)
    img = Draw.MolsToGridImage(mols, molsPerRow=3, highlightAtomLists=[mol.GetSubstructMatch(core) for mol in mols], useSVG=True)

    ## Save the image in the output folder
    image_path = os.path.join(output_folder, f"image_{i}.png")
    cairosvg.svg2png(bytestring=img.data, write_to=image_path)
    break

print(f"Image saved: {image_path}")


#%%
print("######## Loading dataset...")
train_dataset = MoleculeDataset(root="data/split_data", filename="HIV_train_oversampled.csv")
test_dataset = MoleculeDataset(root="data/split_data", filename="HIV_test.csv", test=True)       

# %%
print("######## Loading GNN1 Model...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = GNN1(feature_size=train_dataset[0].x.shape[1])
model = model.to(device)
print(f"######### Number of parameters: {count_parameters(model)}")
print(model)
# %%
# Loss and Optimizer
# Due to imbalance postive and negative label so apply weight in the +ve side < 1 increases precision, > 1 recall
weight = torch.tensor([1,10], dtype=torch.float32).to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight==weight)
optimizer = torch.optim.SGD(model.parameters(), 
                            lr=0.1,
                            momentum=0.9)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# %%
NUM_GRAPHS_PER_BATCH = 256
train_loader = DataLoader(train_dataset,
                          batch_size = NUM_GRAPHS_PER_BATCH, shuffle=True)
test_loader = DataLoader(test_dataset,
                          batch_size = NUM_GRAPHS_PER_BATCH, shuffle=True)

#%%


#%%

def train(epoch, model, train_loader, optimizer, loss_fn):
    #Enumerate over the data
    all_preds = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for _,batch in enumerate(tqdm(train_loader)):
        #Using GPU
        batch.to(device)
        #Reset gradient
        optimizer.zero_grad()
        #passing the node feature and concat the info
        pred = model(batch.x.float(),
                     batch.edge_attr.float,
                     batch.edge_index,
                     batch.batch)
        
        # Calculating the loss and gradients
        loss = loss_fn(torch.squeeze(pred), batch.y.float())
        loss.backward()  
        optimizer.step()  
        # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        all_labels.append(batch.y.cpu().detach().numpy())
    
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels, epoch, "train")
    return running_loss/step
#%%   
def test(epoch, model, test_loader, loss_fn):
    #Enumerate over the data
    all_preds = []
    all_preds_raw = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for batch in test_loader:
        #Using GPU
        batch.to(device)
        pred = model(batch.x.float(),
                    batch.edge_attr.float,
                    batch.edge_index,
                    batch.batch)
        loss = loss_fn(torch.squeeze(pred), batch.y.float())            
        # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        all_preds_raw.append(torch.sigmoid(pred).cpu().detach().numpy())
        all_labels.append(batch.y.cpu().detach().numpy())
    
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    print(all_preds_raw[0][:10])
    print(all_preds[:10])
    print(all_labels[:10])
    calculate_metrics(all_preds, all_labels, epoch, "test")
    log_conf_matrix(all_preds, all_labels, epoch)
    return running_loss/step

#%%

def log_conf_matrix(y_pred, y_true, epoch):
    # Log confusion matrix as image
    cm = confusion_matrix(y_pred, y_true)
    classes = ["0", "1"]
    df_cfm = pd.DataFrame(cm, index = classes, columns = classes)
    plt.figure(figsize = (10,7))
    cfm_plot = sns.heatmap(df_cfm, annot=True, cmap='Blues', fmt='g')
    cfm_plot.figure.savefig(f'data/images/cm_{epoch}.png')
   
    
def calculate_metrics(y_pred, y_true, epoch, type):
    print(f"\n Confusion matrix: \n {confusion_matrix(y_pred, y_true)}")
    print(f"F1 Score: {f1_score(y_true, y_pred)}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    
    try:
        roc = roc_auc_score(y_pred,y_true)
        print(f"ROC AUC: {roc}")
    except:
        print(f"ROC AUC: not definded")
# %%
# Start training
print("###### Start GNN Model training")
best_loss = 1000
early_stopping_counter = 0
for epoch in range(300): 
    if early_stopping_counter <= 10: # = x * 5 
        # Training
        model.train()
        loss = train(epoch, model, train_loader, optimizer, loss_fn)
        print(f"Epoch {epoch} | Train Loss {loss}")
        #mlflow.log_metric(key="Train loss", value=float(loss), step=epoch)

        # Testing
        model.eval()
        if epoch % 5 == 0:
            loss = test(epoch, model, test_loader, loss_fn)
            print(f"Epoch {epoch} | Test Loss {loss}")
           # mlflow.log_metric(key="Test loss", value=float(loss), step=epoch)
            
            # Update best loss
            if float(loss) < best_loss:
                best_loss = loss
                # Save the currently best model 
                #mlflow.pytorch.log_model(model, "model", signature=SIGNATURE)
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

        scheduler.step()
    else:
        print("Early stopping due to no improvement.")
        print([best_loss])
print(f"Finishing training with best test loss: {best_loss}")
print([best_loss])

output_folder = "model_weight"
os.makedirs(output_folder, exist_ok=True)
model_path = os.join.path(output_folder,"model.pth") # Replace with the desired path to save the model
torch.save(model, model_path)

# %%



