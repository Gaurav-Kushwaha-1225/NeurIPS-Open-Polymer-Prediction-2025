import torch
from sklearn.metrics import mean_absolute_error
import joblib
import pandas as pd
import numpy as np
import glob
from train import GNNWithGlobalFeats, mol_to_pyg_data, atom_features, Data, PyGDataLoader

def eval_on_host_train(
    target,
    host_train_csv,
    model_pattern="model_{}_fold*.pt",
    desc_cols_file=None,
    aggregate="mean",
    evaluate=False,
):
    # load host train
    df = pd.read_csv(host_train_csv).reset_index(drop=True)
    n = len(df)
    desc_cols = (
        joblib.load(desc_cols_file)
        if desc_cols_file
        else [c for c in df.columns if c not in ["SMILES", target]]
    )
    # collect models
    model_files = sorted(glob.glob(model_pattern.format(target)))
    assert model_files, "No fold models found"
    all_preds = np.zeros((len(model_files), n), dtype=float)
    for i, mp in enumerate(model_files):
        pkg = torch.load(mp, map_location="cpu")
        # load per-fold scalers (must exist)
        desc_scaler = joblib.load(pkg["scaler_files"]["desc"])
        y_scaler = joblib.load(pkg["scaler_files"]["y"])
        # scale whole-host descriptors using this fold's scaler
        X = df[desc_cols].values.astype(float)
        Xs = desc_scaler.transform(X)
        # build graphs for host rows (order preserved)
        data_list = []
        for idx in range(n):
            smi = df.loc[idx, "SMILES"]
            d = mol_to_pyg_data(smi, global_features=Xs[idx], y=None)
            if d is None:
                # fallback single-node graph
                from rdkit import Chem

                zero = torch.zeros(
                    (1, len(atom_features(Chem.Atom("C")))), dtype=torch.float
                )
                d = Data(
                    x=zero,
                    edge_index=torch.tensor([[0], [0]], dtype=torch.long),
                    edge_attr=torch.zeros((1, 4)),
                    global_feats=torch.tensor(Xs[idx], dtype=torch.float),
                )
            d.orig_idx = torch.tensor(idx, dtype=torch.long)
            data_list.append(d)
        loader = PyGDataLoader(
            data_list, batch_size=64, shuffle=False, num_workers=0
        )
        # instantiate model and load weights
        model = GNNWithGlobalFeats(
            node_in_dim=pkg["node_dim"],
            edge_in_dim=pkg["edge_dim"],
            global_in_dim=pkg["global_dim"],
            gnn_hidden=pkg.get("gnn_hidden", 128),
            n_gnn_layers=pkg.get("n_gnn_layers", 3),
            mlp_hidden=pkg.get("mlp_hidden", 128),
            dropout=pkg.get("dropout", 0.2),
            conv_type=pkg.get("conv_type", "gcn"),
        )
        model.load_state_dict(pkg["state_dict"])
        model.eval()
        preds_fold = np.zeros(n, dtype=float)

        with torch.no_grad():
            for batch in loader:
                batch = batch.to("cpu")
                out = model(batch).detach().cpu().numpy()
                if hasattr(batch, "orig_idx"):
                    idxs = batch.orig_idx.detach().cpu().numpy().ravel()
                    for p, idx in zip(out.tolist(), idxs.tolist()):
                        preds_fold[int(idx)] = p
                else:
                    # fallback sequential
                    pass
        # inverse-scale fold preds to original units
        preds_orig = y_scaler.inverse_transform(
            preds_fold.reshape(-1, 1)
        ).ravel()
        all_preds[i, :] = preds_orig

    # aggregate
    if aggregate == "mean":
        final = all_preds.mean(axis=0)
    else:
        final = all_preds.mean(axis=0)  # extendable to weighted

    if evaluate:
        # compute host-train MAE
        host_mae = mean_absolute_error(df[target].values.astype(float), final)
        print(
            f"Host-train MAE for {target}: {host_mae:.6f} (using {len(model_files)} fold models)"
        )
        return host_mae, final, all_preds
    else:
        return final, all_preds


submission_df = {}
for label in ["Density", "Rg"]:
    host_csv = f"./Datasets/{label}/{label}.csv"
    mae, preds, allp = eval_on_host_train(
        label,
        host_csv,
        model_pattern=f"model_{label}_fold*.pt",
        desc_cols_file=f"desc_cols_{label}.pkl",
        evaluate=True,
    )
    submission_df[label] = preds

print(submission_df)