import os
os.environ['TF_KERAS'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import pickle
import json
from typing import Dict, List, Any
import os
import deepchem as dc
from tqdm import tqdm
import math
from MY_GNN.inference import eval_on_host_train
from NIPS_GNN.inference import predict as nips_predict

class DeepChemTqdmCallback:
    """
    DeepChem-style callback: called as callback(model, current_step).
    Shows a per-epoch tqdm bar (updates once per batch).
    """
    def __init__(self, dataset, batch_size, leave=False):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.leave = leave
        # Try to infer dataset length
        try:
            self.n = len(dataset)
        except Exception:
            y = getattr(dataset, "y", None)
            if hasattr(y, "shape"):
                self.n = int(y.shape[0])
            else:
                self.n = None
        self.steps = None if self.n is None else math.ceil(self.n / self.batch_size)
        self.pbar = None
        self.last_epoch = -1

    def __call__(self, model, current_step):
        """
        Called by DeepChem as callback(model, current_step) after each batch.
        current_step is an integer (global batch count).
        """
        # ensure int
        step = int(current_step)

        # If we can't infer steps_per_epoch, show an indeterminate progress spinner
        if self.steps is None:
            if self.pbar is None:
                self.pbar = tqdm(total=None, desc=f"Step {step}", leave=self.leave)
            else:
                self.pbar.update(1)
            return

        # Determine epoch and batch-within-epoch
        epoch = step // self.steps
        batch_in_epoch = step % self.steps

        # If new epoch, close previous bar and open a new one
        if epoch != self.last_epoch:
            if self.pbar is not None:
                try:
                    self.pbar.close()
                except Exception:
                    pass
            self.pbar = tqdm(total=self.steps, desc=f"Epoch {epoch+1}", leave=self.leave)
            self.last_epoch = epoch
            # Update bar to current batch (handles possible non-1 step jumps)
            # Usually first call in epoch will have batch_in_epoch == 0 -> update by 1
            self.pbar.update(batch_in_epoch + 1)
            return

        # Same epoch: advance by 1 (typical case)
        if self.pbar is not None:
            self.pbar.update(1)

    def close(self):
        """Call after training to ensure bar closed."""
        if self.pbar is not None:
            try:
                self.pbar.close()
            except Exception:
                pass
            self.pbar = None

class PolymerPropertyPredictor:
    def __init__(self):
        self.models = {}
        self.load_models()
        
    def load_models(self):
        # Load different models for different properties
        self.models = {
            'rg': "MY_GNN/trained_models/rg",
            'density': "MY_GNN/trained_models/density",
            'ffv': "NIPS_GNN/trained_models",
            'tg': "NIPS_GNN/trained_models",
            'tc': f"DA_GNN/trained_models/2 layers",
        }
    
    def predict_rg_density(self, SMILES: str) -> Dict[str, float]:
        """Predict Rg and Density using ensemble of models"""
        output_df = pd.DataFrame(columns=["SMILES", "Density", "Rg"])
        output_df["SMILES"] = [SMILES]
        for label in ["Density", "Rg"]:
            host_csv = pd.DataFrame([SMILES], columns=['SMILES'])
            model_dir = self.models[label.lower()]
            preds, allp = eval_on_host_train(
                label,
                host_csv,
                model_pattern=f"{model_dir}/model_{label}_fold*.pt",
                desc_cols_file=f"{model_dir}/desc_cols_{label}.pkl",
                evaluate=False,
            )
            output_df[label] = preds

        return {
            'rg': float(output_df["Rg"].values[0]) if isinstance(output_df["Rg"].values[0], (float, int)) else None,
            'density': float(output_df["Density"].values[0]) if isinstance(output_df["Density"].values[0], (float, int)) else None
        }

    def predict_ffv_tg(self, SMILES: str) -> Dict[str, float]:
        """Predict FFV and Tg using molecular fingerprint approach"""
        pred_df = pd.DataFrame(columns=["SMILES", "FFV", "Tg"])
        pred_df["SMILES"] = [SMILES]
        for target in ['FFV', 'Tg']:
            dict_path = f'{self.models[target.lower()]}/{target.lower()}_dictionaries.pkl'
            smiles_list = [
                SMILES
            ]
            model_path = f'{self.models[target.lower()]}/{target.lower()}_model.pt' # Using trained models
            predictions = nips_predict(smiles_list, target, model_path, dict_path)
            for i, pred in enumerate(predictions):
                if pred is not None:
                    pred_df[target] = pred[0][0][0]
                else:
                    pred_df[target] = None

        return {
            'ffv': float(pred_df["FFV"].values[0]) if isinstance(pred_df["FFV"].values[0], (float, int)) else None,
            'tg': float(pred_df["Tg"].values[0]) if isinstance(pred_df["Tg"].values[0], (float, int)) else None
        }
    
    def predict_tc(self, SMILES: str) -> float | None:
        """Predict Tc using data augmentation model"""
        # Apply your specific preprocessing for Tc
        Restore_MODEL_DIR = self.models['tc']
        smiles = [
            SMILES
        ]
        smiles_df = pd.DataFrame(smiles, columns=['SMILES'])
        
        # Featurizerization
        print("# Featurizerization -> ", end="")
        featurizer = dc.feat.ConvMolFeaturizer()
        smiles_list = smiles_df['SMILES'].tolist()
        molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        featurized_mols = featurizer.featurize(molecules)
        testset = dc.data.NumpyDataset(X=featurized_mols)

        val_pred = []
        print("Predicting -> ", end="")
        for i in range(5):
            MODEL_DIR = Restore_MODEL_DIR + '/' + 'loop' + str(i + 1)
            model = dc.models.GraphConvModel(1, mode="regression", model_dir=MODEL_DIR)
            model.restore()
        
            # Predict
            val_pred.append(model.predict(testset))
        
        print("Done")
        val_pred = sum(val_pred) / len(val_pred)
        
        return float(val_pred[0]) if isinstance(val_pred[0], (float, int)) else None
    
    def predict_all_properties(self, smiles: str) -> Dict[str, float]:
        """Main prediction function"""
        try:
            # Step 2: Property-specific predictions
            rg_density = self.predict_rg_density(smiles)
            ffv_tg = self.predict_ffv_tg(smiles)
            tc = self.predict_tc(smiles)
            
            # Step 3: Combine all predictions
            predictions = {
                'rg': rg_density['rg'],
                'density': rg_density['density'],
                'ffv': ffv_tg['ffv'],
                'tg': ffv_tg['tg'],
                'tc': tc
            }
            
            return predictions
            
        except Exception as e:
            raise ValueError(f"Prediction Error: {str(e)}")

# Global predictor instance
predictor = None

def load_model():
    """Load the model (called once when container starts)"""
    global predictor
    if predictor is None:
        predictor = PolymerPropertyPredictor()
    return predictor

def predict(inputs):
    """Main prediction function for HuggingFace"""
    try:
        # Load model if not loaded
        model = load_model()
        
        # Handle different input formats
        if isinstance(inputs, str):
            smiles = inputs
        elif isinstance(inputs, dict):
            smiles = inputs.get('inputs', inputs.get('smiles', ''))
        elif isinstance(inputs, list) and len(inputs) > 0:
            smiles = inputs[0] if isinstance(inputs[0], str) else inputs[0].get('inputs', '')
        else:
            raise ValueError("Invalid input format")
        
        # Make prediction
        predictions = model.predict_all_properties(smiles)
        
        # Format output
        result = {
            'smiles': smiles,
            'predictions': predictions,
            'properties': {
                'Tg (Glass Transition Temperature)': f"{predictions['tg']:.2f} °C",
                'Tc (Crystallization Temperature)': f"{predictions['tc']:.2f} °C",
                'FFV (Fractional Free Volume)': f"{predictions['ffv']:.4f}",
                'Density': f"{predictions['density']:.3f} g/cm³",
                'Rg (Radius of Gyration)': f"{predictions['rg']:.2f} Å"
            }
        }
        
        return result
        
    except Exception as e:
        return {"error": str(e)}
