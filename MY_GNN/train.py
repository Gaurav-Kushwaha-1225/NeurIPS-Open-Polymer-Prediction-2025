import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import joblib
import pandas as pd
import numpy as np
import os
from rdkit import Chem
import random
import math
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

def compute_weighted_score(per_target_maes: dict, sample_counts: dict):
    """
    per_target_maes: {'Tg': mae_value, 'FFV':..}
    sample_counts: {'Tg': N_tg, ...}
    weights = (1/sqrt(N)) normalized to sum 1
    """
    keys = list(per_target_maes.keys())
    inv_sqrt = np.array([1.0/math.sqrt(sample_counts[k]) for k in keys])
    weights = inv_sqrt / inv_sqrt.sum()
    score = sum(per_target_maes[k] * w for k,w in zip(keys, weights))
    return score, dict(zip(keys, weights))

ATOM_LIST = ["C","H","O","N","S","F","Cl","Br","I","P"]
MAX_DEGREE = 5
def atom_features(atom):
    at = atom.GetSymbol()
    one_hot = [1.0 if at == a else 0.0 for a in ATOM_LIST]
    if not any(one_hot): one_hot.append(1.0)
    else: one_hot.append(0.0)
    charge = [atom.GetFormalCharge()]
    aromatic = [1.0 if atom.GetIsAromatic() else 0.0]
    deg = atom.GetDegree()
    deg_oh = [1.0 if deg==d else 0.0 for d in range(MAX_DEGREE+1)]
    feats = one_hot + charge + aromatic + deg_oh
    return np.array(feats, dtype=np.float32)

def mol_to_pyg_data(smiles, global_features=None, y=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    node_feats = [atom_features(a) for a in mol.GetAtoms()]
    x = torch.tensor(np.vstack(node_feats), dtype=torch.float)
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx(); a2 = bond.GetEndAtomIdx()
        edge_index.append([a1, a2]); edge_index.append([a2, a1])
        bt = bond.GetBondType()
        bond_type = [0.0,0.0,0.0,0.0]
        if bt == Chem.rdchem.BondType.SINGLE: bond_type[0]=1.0
        elif bt == Chem.rdchem.BondType.DOUBLE: bond_type[1]=1.0
        elif bt == Chem.rdchem.BondType.TRIPLE: bond_type[2]=1.0
        elif bt == Chem.rdchem.BondType.AROMATIC: bond_type[3]=1.0
        edge_attr.append(bond_type); edge_attr.append(bond_type)

    if len(edge_index)==0:
        edge_index = [[0,0]]; edge_attr = [[0,0,0,0]]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(np.vstack(edge_attr), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    if global_features is not None:
        gf = torch.tensor(global_features, dtype=torch.float)
        if gf.dim()==1: gf = gf.unsqueeze(0)  # shape (1, D)
        data.global_feats = gf
    else:
        data.global_feats = torch.zeros(1, dtype=torch.float)
    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float)
    return data

def mol_to_scaffold(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, isomericSmiles=False)
    except Exception:
        return None

def scaffold_fold_assignments(df, n_folds=5, smiles_col="SMILES", seed=42):
    random.seed(seed)
    scaffolds = {}
    for idx, smi in enumerate(df[smiles_col].values):
        scaf = mol_to_scaffold(smi)
        if scaf is None: scaf = f"EMPTY_{idx}"
        scaffolds.setdefault(scaf, []).append(idx)
    groups = sorted(scaffolds.items(), key=lambda x: len(x[1]), reverse=True)
    fold_sizes = [0]*n_folds
    fold_assign = np.zeros(len(df), dtype=int)
    for scaf, idxs in groups:
        f = int(np.argmin(fold_sizes))
        for idx in idxs: fold_assign[idx] = f
        fold_sizes[f] += len(idxs)
    return fold_assign

class GNNWithGlobalFeats(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, global_in_dim,
                 gnn_hidden=128, n_gnn_layers=3, mlp_hidden=128, dropout=0.2,
                 conv_type='gcn'):  # NEW: conv_type
        super().__init__()
        self.global_in_dim = global_in_dim
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()  # CHANGED: BatchNorms for node features after conv
        in_dim = node_in_dim
        for _ in range(n_gnn_layers):
            if conv_type == 'gcn':
                self.convs.append(GCNConv(in_dim, gnn_hidden))
            elif conv_type == 'gat':
                # CHANGED: GAT with single head for simplicity
                self.convs.append(GATConv(in_dim, gnn_hidden // 1, heads=1, concat=False))
            else:
                raise ValueError("conv_type must be 'gcn' or 'gat'")
            self.bns.append(nn.BatchNorm1d(gnn_hidden))
            in_dim = gnn_hidden

        self.pool = global_mean_pool
        total_in = gnn_hidden + global_in_dim
        self.mlp = nn.Sequential(
            nn.Linear(total_in, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden//2, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)        # CHANGED
            x = F.relu(x)
            x = self.dropout(x)
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        pooled = self.pool(x, batch)
        gfeat = data.global_feats.to(pooled.dtype).to(pooled.device)
        # same robust handling as before
        if gfeat.dim() == 1:
            if gfeat.numel() == self.global_in_dim:
                gfeat = gfeat.unsqueeze(0).expand(pooled.size(0), -1)
            elif gfeat.numel() == pooled.size(0) * self.global_in_dim:
                gfeat = gfeat.view(pooled.size(0), self.global_in_dim)
            else:
                raise ValueError("Unexpected global_feats shape")
        elif gfeat.dim() == 2:
            if gfeat.size(0) != pooled.size(0) and gfeat.numel() == pooled.size(0) * self.global_in_dim:
                gfeat = gfeat.view(pooled.size(0), self.global_in_dim)
            elif gfeat.size(0) != pooled.size(0):
                gfeat = gfeat.mean(dim=0, keepdim=True).expand(pooled.size(0), -1)

        out = self.mlp(torch.cat([pooled, gfeat], dim=1))
        return out.view(-1)

def train_one_epoch(model, loader, optimizer, device, loss_fn, clip_norm=None):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = loss_fn(pred, batch.y.view(-1))
        loss.backward()
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)  # CHANGED
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

def evaluate_mae_inverse_scaled(model, loader, device, y_scaler):
    """Evaluate MAE but invert target scaling back to original units before MAE."""
    model.eval()
    ys = []
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch).detach().cpu().numpy().tolist()
            y_batch = batch.y.view(-1).detach().cpu().numpy().tolist()
            preds.extend(pred)
            ys.extend(y_batch)
    # inverse transform (y_scaler expects 2D)
    preds = np.array(preds).reshape(-1, 1)
    ys = np.array(ys).reshape(-1, 1)
    preds_orig = y_scaler.inverse_transform(preds).ravel()
    ys_orig = y_scaler.inverse_transform(ys).ravel()
    return mean_absolute_error(ys_orig, preds_orig), ys_orig, preds_orig

def randomized_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    return Chem.MolToSmiles(mol, doRandom=True)

def run_single_train_until_target(csv_path,
                                  target_col,
                                  descriptor_cols,
                                  smiles_col="SMILES",
                                  device='cuda' if torch.cuda.is_available() else 'cpu',
                                  seed=42,
                                  max_epochs=1000,
                                  patience=30,
                                  batch_size=32,
                                  conv_type='gcn',
                                  n_augment_small=1,
                                  clip_grad_norm=5.0,
                                  target_mae=None,
                                  tol_rel=0.05,
                                  tol_abs=1e-6,
                                  verbose=True):
    """
    Train one model on a single train/val split and stop early when validation MAE
    is within tolerance of `target_mae` (or when early-stopping triggers).
    Returns saved artifact paths and the achieved val MAE.
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    df = pd.read_csv(csv_path)
    df = df[df[target_col].notna()].reset_index(drop=True)

    # Create scaffold folds and pick a single validation fold (fold 0)
    fold_assign = scaffold_fold_assignments(df, n_folds=5, smiles_col=smiles_col, seed=seed)
    train_idx = [i for i,f in enumerate(fold_assign) if f != 0]
    val_idx   = [i for i,f in enumerate(fold_assign) if f == 0]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)

    # Fit scalers on TRAIN only
    desc_scaler = StandardScaler().fit(train_df[descriptor_cols].values.astype(float))
    y_scaler = StandardScaler().fit(train_df[[target_col]].values.astype(float))

    # Transform
    X_train = desc_scaler.transform(train_df[descriptor_cols].values.astype(float))
    X_val   = desc_scaler.transform(val_df[descriptor_cols].values.astype(float))
    y_train = y_scaler.transform(train_df[[target_col]].values.astype(float)).ravel()
    y_val   = y_scaler.transform(val_df[[target_col]].values.astype(float)).ravel()

    # Build graph objects
    train_data = []
    for i in range(len(train_df)):
        smi = train_df.loc[i, smiles_col]
        d = mol_to_pyg_data(smi, global_features=X_train[i], y=float(y_train[i]))
        if d is None:
            continue
        d.idx = train_idx[i]
        d.orig_smiles = smi
        train_data.append(d)

    # optional augmentation for small train sets
    if len(train_data) < 2000 and n_augment_small > 0:
        aug_list = []
        for d in train_data:
            for _ in range(n_augment_small):
                rs = randomized_smiles(d.orig_smiles)
                if rs is None: continue
                aug_d = mol_to_pyg_data(rs, global_features=d.global_feats.detach().cpu().numpy().ravel(), y=d.y.item())
                if aug_d is None: continue
                aug_d.idx = d.idx
                aug_d.orig_smiles = rs
                aug_list.append(aug_d)
        if aug_list:
            train_data += aug_list
            if verbose: print(f"Augmented train set with {len(aug_list)} randomized-smiles.")

    val_data = []
    for i in range(len(val_df)):
        smi = val_df.loc[i, smiles_col]
        d = mol_to_pyg_data(smi, global_features=X_val[i], y=float(y_val[i]))
        if d is None:
            continue
        d.idx = val_idx[i]
        d.orig_smiles = smi
        val_data.append(d)

    # Dataloaders
    train_loader = PyGDataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = PyGDataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)

    # model
    node_dim = train_data[0].x.shape[1]
    edge_dim = train_data[0].edge_attr.shape[1] if hasattr(train_data[0], 'edge_attr') else 0
    global_dim = train_data[0].global_feats.shape[0] if train_data[0].global_feats.dim()==1 else train_data[0].global_feats.shape[1]

    model = GNNWithGlobalFeats(node_in_dim=node_dim, edge_in_dim=edge_dim, global_in_dim=global_dim,
                               gnn_hidden=128, n_gnn_layers=3, mlp_hidden=128, dropout=0.2,
                               conv_type=conv_type).to(device)

    loss_fn = nn.SmoothL1Loss()
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7, verbose=verbose, min_lr=1e-6)

    best_val_mae = float('inf')
    best_state = None
    no_improve = 0
    best_epoch = -1

    # target stopping thresholds
    stop_enabled = (target_mae is not None)
    if stop_enabled:
        tol = max(tol_rel * float(target_mae), tol_abs)
        if verbose:
            print(f"Target MAE {target_mae} with tolerance {tol} (relative tol {tol_rel}, abs tol {tol_abs})")

    for epoch in range(1, max_epochs+1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn, clip_norm=clip_grad_norm)
        val_mae, ys_orig, preds_orig = evaluate_mae_inverse_scaled(model, val_loader, device, y_scaler)
        scheduler.step(val_mae)

        if val_mae < best_val_mae - 1e-6:
            best_val_mae = val_mae
            best_epoch = epoch
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"Epoch {epoch:04d} | train_loss {train_loss:.6f} | val_mae_orig {val_mae:.6f} | best_val {best_val_mae:.6f}")

        # stop if we reached target MAE within tolerance
        if stop_enabled and abs(val_mae - float(target_mae)) <= tol:
            if verbose:
                print(f"Stopping at epoch {epoch} because val_mae {val_mae:.6f} is within tolerance of target {target_mae}.")
            break

        # normal early stopping
        if no_improve >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch} (best_epoch {best_epoch} | best_val_mae {best_val_mae:.6f})")
            break

    # restore best
    if best_state is not None:
        model.load_state_dict({k: best_state[k].to(device) for k in best_state})

    # final metric
    val_mae_final, ys_orig, preds_orig = evaluate_mae_inverse_scaled(model, val_loader, device, y_scaler)
    if verbose:
        print(f"Final val MAE (orig scale): {val_mae_final:.6f} (best_epoch {best_epoch})")

    # save model + scalers (single-model artifacts)
    base = f"{target_col}_single"
    pkg = {
        'state_dict': {k: v.cpu().clone() for k, v in model.state_dict().items()},
        'node_dim': node_dim, 'edge_dim': edge_dim, 'global_dim': global_dim,
        'gnn_hidden': 128, 'n_gnn_layers': 3, 'mlp_hidden': 128, 'dropout': 0.2,
        'conv_type': conv_type, 'val_mae_orig': float(val_mae_final)
    }
    model_path = f"model_{base}.pt"
    desc_path = f"desc_scaler_{base}.pkl"
    y_path = f"y_scaler_{base}.pkl"
    joblib.dump(desc_scaler, desc_path)
    joblib.dump(y_scaler, y_path)
    torch.save(pkg, model_path)
    if verbose:
        print(f"Saved model -> {model_path}; scalers -> {desc_path}, {y_path}")

    return {
        'model_path': model_path,
        'desc_scaler_path': desc_path,
        'y_scaler_path': y_path,
        'val_mae': val_mae_final,
        'epoch': best_epoch,
        'preds_val': preds_orig,
        'y_val': ys_orig
    }

if __name__ == "__main__":
    dataset_dir = "/kaggle/input/augmented-polymer-data/results"
    targets_to_train = {
        # 'Tg': 39.6794,
        # 'FFV': 0.0042,
        # 'Tc': 0.0211,
        'Density': 0.0182,
        'Rg': 1.1638
    }

    results = {}
    for t, goal in targets_to_train.items():
        csv_path = os.path.join(dataset_dir, f"{t}_data.csv")
        print(f"\n=== Training single-model for {t} aiming MAE ~ {goal} ===")
        df_tmp = pd.read_csv(csv_path)
        descriptor_cols = df_tmp.drop(columns=[t, 'SMILES']).columns.tolist()
        joblib.dump(descriptor_cols, f"desc_cols_{t}_single.pkl")
        res = run_single_train_until_target(csv_path=csv_path,
                                            target_col=t,
                                            descriptor_cols=descriptor_cols,
                                            smiles_col="SMILES",
                                            device='cuda' if torch.cuda.is_available() else 'cpu',
                                            seed=42,
                                            max_epochs=1000,
                                            patience=40,
                                            batch_size=32,
                                            conv_type='gcn',
                                            n_augment_small=1,
                                            clip_grad_norm=5.0,
                                            target_mae=goal,
                                            tol_rel=0.05,
                                            tol_abs=1e-6,
                                            verbose=True)
        results[t] = res
        print(f"-> {t} achieved val MAE {res['val_mae']:.6f} (target {goal})")