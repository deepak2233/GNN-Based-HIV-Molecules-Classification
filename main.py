import argparse
import os
import torch
import pandas as pd
from torch_geometric.data import DataLoader
from dataset_featurizer import MoleculeDataset
from model.GNN1 import GNN1
from model.GNN2 import GNN2
from model.GNN3 import GNN3
from utils import calculate_metrics, plot_confusion_matrix, save_checkpoint
from tqdm import tqdm
import optuna

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.batch)
        out = out.squeeze()
        loss = loss_fn(out, batch.y.float())
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = (torch.sigmoid(out) > 0.5).float()
        all_preds.append(preds.detach().cpu())
        all_labels.append(batch.y.detach().cpu())
        
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    metrics = calculate_metrics(all_preds, all_labels)
    
    return total_loss / len(loader), metrics

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    for batch in tqdm(loader, desc="Evaluating"):
        batch = batch.to(device)
        out = model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.batch)
        out = out.squeeze()
        loss = loss_fn(out, batch.y.float())
        
        total_loss += loss.item()
        probs = torch.sigmoid(out)
        all_probs.append(probs.detach().cpu())
        all_preds.append((probs > 0.5).float().detach().cpu())
        all_labels.append(batch.y.detach().cpu())
        
    all_preds = torch.cat(all_preds).numpy()
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    metrics = calculate_metrics(all_preds, all_labels, y_prob=all_probs)
    
    return total_loss / len(loader), metrics, all_preds, all_labels

def main():
    parser = argparse.ArgumentParser(description='GNN HIV Classification Pipeline')
    parser.add_argument('--mode', type=str, choices=['train', 'optimize', 'test'], default='train')
    parser.add_argument('--model_type', type=str, choices=['GNN1', 'GNN2', 'GNN3'], default='GNN1')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--train_data', type=str, default='data/split_data/HIV_train.csv')
    parser.add_argument('--test_data', type=str, default='data/split_data/HIV_test.csv')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--quick_test', action='store_true', help='Run on a small subset for verification')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    
    # Load Datasets
    print("Loading datasets...")
    train_dataset = MoleculeDataset(root="data/split_data", filename=os.path.basename(args.train_data))
    test_dataset = MoleculeDataset(root="data/split_data", filename=os.path.basename(args.test_data), test=True)
    
    if args.quick_test:
        train_dataset = train_dataset[:100]
        test_dataset = test_dataset[:50]
        print("Running in quick_test mode (subsetting data).")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize Model
    feature_size = train_dataset[0].x.shape[1]
    edge_feature_size = train_dataset[0].edge_attr.shape[1]
    
    if args.model_type == 'GNN1':
        model = GNN1(feature_size=feature_size).to(device)
    elif args.model_type == 'GNN2':
        model = GNN2(feature_size=feature_size).to(device)
    else:
        model = GNN3(feature_size=feature_size, edge_feature_size=edge_feature_size).to(device)
        
    # Loss and Optimizer
    # High-imbalance handling: Active class weighting
    # In HIV dataset, active (1) is much rarer than inactive (0)
    pos_weight = torch.tensor([15.0]).to(device) # Adjust based on ratio ~1:26, 15 is a good start
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    if args.mode == 'train':
        best_f1 = -1.0
        for epoch in range(args.epochs):
            train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            test_loss, test_metrics, y_pred, y_true = evaluate(model, test_loader, loss_fn, device)
            
            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"Train Loss: {train_loss:.4f} | Train F1: {train_metrics['f1_score']:.4f}")
            print(f"Test Loss: {test_loss:.4f} | Test F1: {test_metrics['f1_score']:.4f}")
            
            if test_metrics['f1_score'] > best_f1:
                best_f1 = test_metrics['f1_score']
                save_checkpoint(model, optimizer, epoch, test_loss, os.path.join(args.output_dir, "best_model.pth"))
                plot_confusion_matrix(y_true, y_pred, os.path.join(args.output_dir, "confusion_matrix.png"))
                
    elif args.mode == 'test':
        checkpoint_path = os.path.join(args.output_dir, "best_model.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded best model.")
        
        test_loss, test_metrics, y_pred, y_true = evaluate(model, test_loader, loss_fn, device)
        print("Test Metrics:")
        for k, v in test_metrics.items():
            print(f"{k}: {v:.4f}")
        plot_confusion_matrix(y_true, y_pred, os.path.join(args.output_dir, "test_confusion_matrix.png"))

    elif args.mode == 'optimize':
        def objective(trial):
            lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
            
            # Simplified training for optimization
            trial_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            trial_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            for _ in range(5): # Fewer epochs for tuning
                train_one_epoch(model, trial_loader, trial_optimizer, loss_fn, device)
                
            _, test_metrics, _, _ = evaluate(model, test_loader, loss_fn, device)
            return test_metrics['f1_score']

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)
        print("Best params:", study.best_params)

if __name__ == "__main__":
    main()
