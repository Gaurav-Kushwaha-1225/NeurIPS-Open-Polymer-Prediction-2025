import torch
from sklearn.metrics import mean_absolute_error
import joblib
import pandas as pd
import numpy as np
import glob
import os
from MY_GNN.train import GNNWithGlobalFeats, mol_to_pyg_data, atom_features, Data, PyGDataLoader
from tqdm import tqdm
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetAtomPairGenerator, GetTopologicalTorsionGenerator
from rdkit.Chem import MACCSkeys
from rdkit import Chem
import networkx as nx
from rdkit.Chem import Descriptors, rdmolops
from rdkit.Chem.Descriptors import MolWt, MolLogP
from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcNumRotatableBonds

required_descriptors = {'graph_diameter','num_cycles','avg_shortest_path','MolWt', 'LogP', 'TPSA', 'RotatableBonds', 'NumAtoms', 'SMILES'}

filters = {
    'Tg': list(set(['deg_mean', 'FractionCSP3', 'num_cycles', 'RingCount', 'HallKierAlpha', 'SMR_VSA7', 'BertzCT', 'ring_size_6', 'fr_benzene', 'NumAromaticCarbocycles', 'NumAromaticRings', 'SlogP_VSA6', 'SlogP_VSA1', 'betw_mean', 'VSA_EState6', 'BalabanJ', 'Chi4n', 'FP_446', 'PEOE_VSA14', 'Chi3n', 'AvgIpc', 'FP_489', 'Chi1', 'HeavyAtomCount', 'NumHeterocycles', 'FP_485', 'fr_bicyclic', 'SMR_VSA10', 'FP_537', 'VSA_EState2', 'FP_539', 'FP_529', 'HeavyAtomMolWt', 'LabuteASA', 'ring_size_5', 'FP_505', 'NumAmideBonds', 'MolMR', 'FP_80', 'FP_195', 'FP_310', 'fr_amide', 'FP_509', 'FP_378', 'ExactMolWt', 'MolWt', 'FP_211', 'Chi2n', 'FP_266', 'FP_379', 'FP_207', 'FP_504', 'FP_203', 'NumAtoms', 'FP_199', 'FP_519', 'FP_123', 'FP_278', 'FpDensityMorgan1', 'FP_119', 'fr_imide', 'FP_279', 'FP_223', 'betw_std', 'FP_231', 'FP_219', 'FP_251', 'NumValenceElectrons', 'FP_480', 'Chi0n', 'FP_517', 'FP_255', 'Chi0', 'FP_522', 'FP_528', 'FP_526', 'FpDensityMorgan2', 'FP_354', 'Chi1n', 'FP_459', 'FP_547', 'FP_476', 'Chi0v', 'FP_210', 'FP_516', 'FP_382', 'FP_215', 'FP_243', 'FP_521', 'FP_227', 'NumAliphaticHeterocycles', 'FP_469', 'FP_467', 'FP_342', 'FP_549', 'FP_357', 'FP_494', 'FP_194', 'FP_546', 'FP_302']).union(required_descriptors)),

    'FFV': list(set(['MolLogP', 'LogP', 'Chi3v', 'Chi2v', 'Chi4v', 'Chi4n', 'Chi3n', 'VSA_EState6', 'SMR_VSA7', 'Chi1v', 'Chi2n', 'MolMR', 'Chi1n', 'Chi0v', 'SlogP_VSA6', 'BertzCT', 'PEOE_VSA14', 'Chi0n', 'LabuteASA', 'EState_VSA8', 'BalabanJ', 'Ipc', 'Chi1', 'deg_mean', 'VSA_EState8', 'MolWt', 'ExactMolWt', 'SMR_VSA9', 'HeavyAtomMolWt', 'Chi0', 'SMR_VSA6', 'EState_VSA5', 'FpDensityMorgan3', 'Kappa1', 'AvgIpc', 'FpDensityMorgan2', 'SlogP_VSA8', 'HallKierAlpha', 'FP_39', 'avg_shortest_path', 'SMR_VSA1', 'SlogP_VSA5', 'betw_mean', 'TPSA', 'FpDensityMorgan1', 'lap_eig_6', 'qed', 'lap_eig_7', 'lap_eig_8', 'RingCount', 'NumValenceElectrons', 'NumAromaticRings', 'lap_eig_5', 'num_cycles', 'EState_VSA7', 'Kappa2', 'NumAtoms', 'ring_size_6', 'betw_std', 'lap_eig_4', 'lap_eig_3', 'HeavyAtomCount', 'fr_benzene', 'NumHDonors', 'NumAromaticCarbocycles', 'PEOE_VSA7', 'SlogP_VSA2', 'SlogP_VSA3', 'NHOHCount', 'NOCount', 'fr_NH1', 'EState_VSA4', 'FP_515', 'MaxEStateIndex', 'MaxAbsEStateIndex', 'lap_eig_2', 'PEOE_VSA6', 'VSA_EState5', 'EState_VSA6', 'FractionCSP3', 'EState_VSA3', 'Phi', 'FP_535', 'NumHAcceptors', 'SlogP_VSA4', 'fr_C_O', 'FP_446', 'FP_488', 'SMR_VSA10', 'PEOE_VSA9', 'fr_C_O_noCOO', 'FP_125', 'FP_474', 'SlogP_VSA7', 'graph_diameter', 'SlogP_VSA12', 'FP_507', 'fr_bicyclic', 'MinAbsEStateIndex', 'deg_std']).union(required_descriptors)),

    'Tc': list(set(['deg_std', 'fr_unbrch_alkane', 'FP_287', 'FP_286', 'betw_mean', 'avg_shortest_path', 'Kappa3', 'graph_diameter', 'FP_285', 'FP_187', 'FP_191', 'FP_139', 'VSA_EState7', 'FpDensityMorgan3', 'FP_143', 'FpDensityMorgan2', 'FP_171', 'Phi', 'FpDensityMorgan1', 'FP_167', 'FP_175', 'qed', 'FP_163', 'FP_131', 'FP_531', 'FP_502', 'FP_142', 'Kappa2', 'FP_135', 'FP_179', 'SlogP_VSA5', 'FP_513', 'FP_134', 'SMR_VSA5', 'FP_93', 'FP_138', 'FP_130', 'RotatableBonds', 'NumRotatableBonds', 'FP_182', 'FP_162', 'FP_518', 'FP_491', 'FP_174', 'FP_496', 'FP_284', 'FP_141', 'FP_475', 'lap_eig_7', 'FP_154', 'lap_eig_8', 'FP_512', 'FP_453', 'FP_256', 'FP_488', 'deg_max', 'FP_170', 'FP_137', 'FP_190', 'FP_17', 'FP_466', 'FP_535', 'FP_474', 'fr_NH1', 'lap_eig_6', 'Chi3n', 'FP_133', 'PEOE_VSA6', 'FP_178', 'FP_186', 'Chi1n', 'FP_450', 'FP_102', 'FP_508', 'EState_VSA5', 'NumHDonors', 'NumAtomStereoCenters', 'NumUnspecifiedAtomStereoCenters', 'NHOHCount', 'FP_183', 'Chi2n', 'Chi3v', 'FP_89', 'SPS', 'betw_max', 'AvgIpc', 'Chi4n', 'Chi1v', 'FP_478', 'lap_eig_5', 'FP_484', 'SMR_VSA3', 'FP_192', 'FP_166', 'Kappa1', 'FP_495', 'FP_526', 'fr_halogen', 'FP_153', 'FP_28']).union(required_descriptors)),

    'Density': list(set(['SMR_VSA5', 'VSA_EState8', 'VSA_EState7', 'SlogP_VSA5', 'SMR_VSA10', 'FractionCSP3', 'EState_VSA5', 'SlogP_VSA12', 'VSA_EState10', 'fr_unbrch_alkane', 'NumRotatableBonds', 'RotatableBonds', 'FP_119', 'FP_513', 'PEOE_VSA8', 'Kappa3', 'PEOE_VSA7', 'FP_180', 'FP_472', 'FP_428', 'Kappa2', 'FP_80', 'Phi', 'FP_539', 'FP_512', 'FP_531', 'EState_VSA7', 'FP_537', 'FP_502', 'FP_98', 'NumHAcceptors', 'Chi1n', 'MaxAbsEStateIndex', 'MaxEStateIndex', 'PEOE_VSA14', 'FP_500', 'MolLogP', 'LogP', 'FP_465', 'MinAbsEStateIndex', 'Chi2n', 'SlogP_VSA7', 'FP_176', 'avg_shortest_path', 'EState_VSA4', 'FP_181', 'lap_eig_5', 'Chi0n', 'HallKierAlpha', 'PEOE_VSA5', 'qed', 'graph_diameter', 'FP_186', 'betw_mean', 'FP_287', 'FP_179', 'lap_eig_4', 'FP_134', 'Chi3n', 'NOCount', 'fr_C_S', 'FP_131', 'FP_177', 'FP_166', 'FP_127', 'FP_162', 'FP_191', 'FP_143', 'Chi4n', 'TPSA', 'lap_eig_3', 'Chi1v', 'SlogP_VSA6', 'FP_178', 'FP_457', 'FP_139', 'FP_163', 'SMR_VSA7', 'SlogP_VSA11', 'SlogP_VSA3', 'FP_183', 'Chi0', 'FP_137', 'ring_size_6', 'FP_138', 'fr_benzene', 'NumAromaticCarbocycles', 'FP_420', 'NumAromaticRings', 'NumAromaticHeterocycles', 'FP_492', 'FP_169', 'FP_284', 'Chi1', 'FP_141', 'FP_35', 'FP_182', 'FP_521', 'EState_VSA3', 'FP_135']).union(required_descriptors)),

    'Rg': list(set(['FP_93', 'SlogP_VSA7', 'PEOE_VSA14', 'qed', 'FP_544', 'VSA_EState8', 'FP_499', 'SlogP_VSA1', 'fr_unbrch_alkane', 'FP_42', 'EState_VSA4', 'FP_192', 'FP_508', 'FP_520', 'lap_eig_8', 'Phi', 'FP_155', 'NumAtomStereoCenters', 'NumUnspecifiedAtomStereoCenters', 'avg_shortest_path', 'FP_17', 'FP_317', 'lap_eig_7', 'FP_73', 'VSA_EState7', 'FP_224', 'fr_ester', 'graph_diameter', 'Kappa2', 'NumAmideBonds', 'fr_NH1', 'FP_191', 'fr_amide', 'FP_286', 'Kappa3', 'FP_159', 'FP_488', 'FP_33', 'deg_std', 'FP_280', 'FP_364', 'FP_287', 'EState_VSA5', 'SlogP_VSA5', 'FP_515', 'TPSA', 'FP_151', 'SMR_VSA5', 'FP_498', 'NOCount', 'betw_mean', 'RotatableBonds', 'NumRotatableBonds', 'FP_273', 'SMR_VSA3', 'FP_163', 'FP_134', 'FP_478', 'FP_138', 'FP_187', 'FP_137', 'FP_252', 'VSA_EState3', 'FP_171', 'FP_175', 'lap_eig_6', 'NHOHCount', 'Chi4v', 'FpDensityMorgan1', 'FP_182', 'FP_526', 'FP_167', 'FP_486', 'FP_142', 'FP_316', 'AvgIpc', 'MolLogP', 'LogP', 'FP_183', 'FP_130', 'FP_102', 'FP_1', 'FP_115', 'SMR_VSA10', 'Chi4n', 'FP_24', 'FP_533', 'NumHDonors', 'FP_193', 'FP_147', 'FP_38', 'Chi3n', 'FP_249', 'FP_453', 'FP_535', 'FP_492', 'Chi3v', 'FP_240', 'FP_501', 'FP_139']).union(required_descriptors))
}

def smiles_to_combined_fingerprints_with_descriptors(smiles_list, radius=2, n_bits=128):
    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
    atom_pair_gen = GetAtomPairGenerator(fpSize=n_bits)
    torsion_gen = GetTopologicalTorsionGenerator(fpSize=n_bits)

    fingerprints = []
    descriptors = []
    valid_smiles = []
    invalid_indices = []

    for i, smiles in tqdm(enumerate(smiles_list), total=len(smiles_list), desc="🔬 Data Augmentation"):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Fingerprints
            morgan_fp = generator.GetFingerprint(mol)
            atom_pair_fp = atom_pair_gen.GetFingerprint(mol)
            torsion_fp = torsion_gen.GetFingerprint(mol)
            maccs_fp = MACCSkeys.GenMACCSKeys(mol)

            combined_fp = np.concatenate([
                np.array(morgan_fp),
                np.array(atom_pair_fp),
                np.array(torsion_fp),
                np.array(maccs_fp)
            ])
            fingerprints.append(combined_fp)

            # RDKit Descriptors
            descriptor_values = {}
            for name, func in Descriptors.descList:
                try:
                    descriptor_values[name] = func(mol)
                except:
                    descriptor_values[name] = None

            # Specific descriptors
            descriptor_values['MolWt'] = MolWt(mol)
            descriptor_values['LogP'] = MolLogP(mol)
            descriptor_values['TPSA'] = CalcTPSA(mol)
            descriptor_values['RotatableBonds'] = CalcNumRotatableBonds(mol)
            descriptor_values['NumAtoms'] = mol.GetNumAtoms()
            descriptor_values['SMILES'] = smiles

            # Graph-based features
            try:
                adj = rdmolops.GetAdjacencyMatrix(mol)
                G = nx.from_numpy_array(adj)

                if nx.is_connected(G):
                    descriptor_values['graph_diameter'] = nx.diameter(G)
                    descriptor_values['avg_shortest_path'] = nx.average_shortest_path_length(G)
                else:
                    descriptor_values['graph_diameter'] = 0
                    descriptor_values['avg_shortest_path'] = 0

                cycles = nx.cycle_basis(G)
                descriptor_values['num_cycles'] = len(list(cycles))
                sizes = [len(c) for c in cycles]
                for k in range(3, 9):
                    descriptor_values[f'ring_size_{k}'] = sizes.count(k)
            except:
                descriptor_values['graph_diameter'] = None
                descriptor_values['avg_shortest_path'] = None
                descriptor_values['num_cycles'] = None
                for k in range(3, 9):
                    descriptor_values[f'ring_size_{k}'] = None
                    
            # Compute Centralities
            adj = rdmolops.GetAdjacencyMatrix(mol)
            G = nx.from_numpy_array(adj)
            deg = dict(nx.degree(G))
            bc = nx.betweenness_centrality(G)
            cc = nx.clustering(G)
            for label, metric in [('deg', deg), ('betw', bc), ('clust', cc)]:
                vals = np.array(list(metric.values()), dtype=float)
                descriptor_values[f'{label}_mean'] = vals.mean()
                descriptor_values[f'{label}_std'] = vals.std()
                descriptor_values[f'{label}_max'] = vals.max()
                    
            # Compute Spectral
            adj = rdmolops.GetAdjacencyMatrix(mol)
            G = nx.from_numpy_array(adj)
            L = nx.normalized_laplacian_matrix(G).toarray()
            eigs = np.linalg.eigvals(L)
            eigs = np.sort(eigs.real)
            for i in range(min(k, len(eigs))):
                descriptor_values[f'lap_eig_{i+1}'] = eigs[i]
            for i in range(len(eigs), k):
                descriptor_values[f'lap_eig_{i+1}'] = 0.0

            descriptor_values['Ipc'] = np.log10(descriptor_values['Ipc'])
            descriptor_values['lap_eig_1'] = np.sign(descriptor_values['lap_eig_1']) * np.log10(np.abs(descriptor_values['lap_eig_1']) + 1e-20)
            
            ###
            descriptors.append(descriptor_values)
            valid_smiles.append(smiles)
        else:
            fingerprints.append(np.zeros(n_bits * 3 + 167))
            descriptors.append(None)
            valid_smiles.append(None)
            invalid_indices.append(i)

    return np.array(fingerprints), descriptors, valid_smiles, invalid_indices

def eval_on_host_train(
    target,
    host_train_csv,
    desc_cols_file,
    model_pattern="model_{}_fold*.pt",
    aggregate="mean",
    evaluate=False,
):
    # load host test
    test_smiles = host_train_csv['SMILES'].tolist()
    
    fingerprints, descriptors, valid_smiles, invalid_indices = smiles_to_combined_fingerprints_with_descriptors(test_smiles, radius=2, n_bits=128)
    
    X = pd.DataFrame(descriptors)
    X = X.drop(['BCUT2D_MWLOW','BCUT2D_MWHI','BCUT2D_CHGHI','BCUT2D_CHGLO','BCUT2D_LOGPHI','BCUT2D_LOGPLOW','BCUT2D_MRLOW','BCUT2D_MRHI','MinAbsPartialCharge','MaxPartialCharge','MinPartialCharge','MaxAbsPartialCharge',],axis=1)
    selected = filters[target]
    X = X.filter(items=selected)

    fp_df = pd.DataFrame(fingerprints, columns=[f'FP_{i}' for i in range(fingerprints.shape[1])])    
    print(fp_df.shape)

    fp_df.reset_index(drop=True, inplace=True)
    X.reset_index(drop=True, inplace=True)
    X = pd.concat([X, fp_df], axis=1)

    print(f"After concat: {X.shape}")
    df = X
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
        model_dir = os.path.dirname(mp)
        desc_scaler_path = pkg["scaler_files"]["desc"]
        y_scaler_path = pkg["scaler_files"]["y"]
        if not os.path.isabs(desc_scaler_path):
            desc_scaler_path = os.path.join(model_dir, desc_scaler_path)
        if not os.path.isabs(y_scaler_path):
            y_scaler_path = os.path.join(model_dir, y_scaler_path)
        desc_scaler = joblib.load(desc_scaler_path)
        y_scaler = joblib.load(y_scaler_path)
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
        if target in df.columns:
            host_mae = mean_absolute_error(df[target].values.astype(float), final)
            print(
                f"Host-train MAE for {target}: {host_mae:.6f} (using {len(model_files)} fold models)"
            )
            return host_mae, final, all_preds
        else:
            print(f"Target column '{target}' not found, skipping MAE calculation.")
            return None, final, all_preds
    else:
        return final, all_preds


if __name__ == "__main__":
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