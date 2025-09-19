import os
import torch
from collections import defaultdict
from rdkit import Chem
from train import MolecularGraphNeuralNetwork, Tester
import preprocess as pp
import pickle
import tqdm
import numpy as np

task = 'regression'
radius = 1
dim = 50
layer_hidden = 6
layer_output = 6
N_fingerprints = 651  # This should match the model's expected input size

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses a GPU!')
else:
    device = torch.device('cpu')
    print('The code uses a CPU...')


def predict(smiles_list, model_path, dict_path):
    # Reinitialize the model structure
    model = MolecularGraphNeuralNetwork(N_fingerprints, dim, layer_hidden, layer_output).to(device)
    
    # Load the saved state_dict
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Load the dictionaries used during training
    with open(dict_path, 'rb') as f:
        atom_dict, bond_dict, fingerprint_dict, edge_dict = pickle.load(f)

    predictions = []
    tester = Tester(model)
    failed = 0
    failed_smiles = []

    for smiles in tqdm.tqdm(smiles_list, total=len(smiles_list), desc=f"Predicting SMILES"):
        try:
            mol = Chem.MolFromSmiles(smiles)
            smiles = Chem.MolToSmiles(mol, canonical=True)
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            atoms = pp.create_atoms(mol, atom_dict)
            molecular_size = len(atoms)
            i_jbond_dict = pp.create_ijbonddict(mol, bond_dict)
            fingerprints = pp.extract_fingerprints(radius, atoms, i_jbond_dict,
                                                   fingerprint_dict, edge_dict)
            adjacency = Chem.GetAdjacencyMatrix(mol)
    
            # Convert to PyTorch tensors
            fingerprints = torch.LongTensor(fingerprints).to(device)
            adjacency = torch.FloatTensor(adjacency).to(device)
    
            dataset = [(fingerprints, adjacency, molecular_size, torch.FloatTensor([[float(0)]]).to(device))]
            predictions.append(tester.predict_regressor(dataset))

        except Exception as e:
            failed += 1
            failed_smiles += [smiles]
            predictions.append(None)

    return predictions

# Example usage
if __name__ == "__main__":
    dict_path = './NIPS_GNN/trained_models/tg_dictionaries.pkl' # Using trained models
    smiles_list = [
        "*Oc1ccc(C=NN=Cc2ccc(Oc3ccc(C(c4ccc(*)cc4)(C(F)(F)F)C(F)(F)F)cc3)cc2)cc1",
        "*Oc1ccc(C(C)(C)c2ccc(Oc3ccc(C(=O)c4cccc(C(=O)c5ccc(*)cc5)c4)cc3)cc2)cc1",
        "*c1cccc(OCCCCCCCCOc2cccc(N3C(=O)c4ccc(-c5cccc6c5C(=O)N(*)C6=O)cc4C3=O)c2)c1"
    ]
    model_path = './NIPS_GNN/trained_models/model.pt' # Using trained models

    predictions = predict(smiles_list, model_path, dict_path)
    for i, pred in enumerate(predictions):
        if pred is not None:
            print(f"{smiles_list[i]}: {pred[0][0][0]}")
        else:
            print(f"Prediction failed for SMILES {smiles_list[i]}")