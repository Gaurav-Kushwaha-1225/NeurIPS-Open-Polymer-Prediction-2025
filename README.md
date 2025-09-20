# NeurIPS Open Polymer Prediction 2025 - Complete Solution Documentation

## Competition Overview

The NeurIPS Open Polymer Prediction 2025 competition challenged participants to predict five key polymer properties from SMILES (Simplified Molecular Input Line Entry System) representations:
- **Tg**: Glass Transition Temperature
- **Tc**: Crystallization Temperature  
- **FFV**: Fractional Free Volume
- **Density**: Polymer Density
- **Rg**: Radius of Gyration

The evaluation metric was weighted Mean Absolute Error (wMAE), with FFV having roughly 10 times the weight of other properties, making it important in the final score.

## My Approach and Solution Architecture

After not getting promising results from Transformers-based and GB-based models, I developed a comprehensive multi-model approach, using different Graph Neural Network (GNN) architectures optimized for different polymer properties based on their unique characteristics and data distributions.

### Property-Specific Model Selection

#### 1. **Rg and Density** → MyGNN (Custom Implementation)
- **Location**: `/MY_GNN/`
- **Architecture**: Custom Graph Neural Network designed specifically for polymer molecular graphs
- **Design Philosophy**: These properties are directly related to molecular structure and spatial arrangements, requiring a custom approach that could capture geometric and topological features effectively
- **Key Features**:
  - Custom message passing mechanism
  - Specialized node and edge feature representations

#### 2. **FFV and Tg** → MolecularGNN_SMILES 
- **Location**: `/NIPS_GNN/`
- **Base Repository**: [masashitsubaki/molecularGNN_smiles](https://github.com/masashitsubaki/molecularGNN_smiles)
- **Architecture**: Graph Neural Network based on learning representations of r-radius subgraphs (molecular fingerprints)
- **Design Philosophy**: FFV and Tg are thermal and mechanical properties that correlate well with local chemical environments and substructural patterns
- **Key Features**:
  - Fingerprints representation learning
  - Proven performance on molecular property prediction tasks

#### 3. **Tc** → DataAugmentation4SmallData (Modified)
- **Location**: `/DA_GNN/`  
- **Base Repository**: [hkqiu/DataAugmentation4SmallData](https://github.com/hkqiu/DataAugmentation4SmallData)
- **Architecture**: Neural network with data augmentation techniques for small datasets
- **Modifications Made**:
  - Adjusted layer sizes for deeper polymer-specific features
  - Modified augmentation strategies for chemical data

## Technical Implementation Details

### Data Preprocessing
- **Input Format**: SMILES strings representing polymer structures
- **Graph Construction**: Molecular graphs with atoms as nodes and bonds as edges
- **Feature Engineering** using `rdkit` and `networkx` modules:
  - Atomic features (element type, hybridization, formal charge, etc.) and Bond features (bond type, conjugation, ring membership, etc.)
  - Global molecular descriptors where applicable

### Model Training Strategy
- **Cross-Validation**: 5-fold cross-validation for robust performance estimation
- **Optimization**: Adam optimizer with learning rate scheduling
- **Loss Function**: Mean Absolute Error (MAE) to match competition metric
- **Early Stopping**: Implemented to prevent overfitting

### Performance Achieved
- **Best Public LB Score**: 0.065 wMAE, which later scored **0.083** wMAE on the Private LB, among the [top 10 performers](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/leaderboard).
- **Model Type**: Ensemble of the three GNN approaches

## Repository Structure

```
NeurIPS-Open-Polymer-Prediction-2025/
├── MY_GNN/                      # Custom GNN for Rg and Density
│   ├── train.py               # Training Script
│   ├── inference.py           # Inference Script
│   └── trained_models/        # Trained Models
├── NIPS_GNN/                    # MolecularGNN for FFV and Tg
├── DA_GNN/                      # Data Augmentation GNN for Tc
├── notebooks/                 # Jupyter submitted notebooks
├── Datasets/                    # Competition Datasets and External ones as well, along with training scripts
└── README.md                  # This file
```

## What Went Wrong: The Final Day Mistake

Despite achieving a strong public leaderboard score of 0.065 with my GNN ensemble, I made a critical error on the final submission day that cost me the competition.

On the last day of the competition, influenced by discussion threads suggesting that models performing poorly on the public leaderboard (which used only ~8% of test data) might perform better on the private leaderboard (remaining ~92%), I decided to submit a different, inferior model instead of my best-performing GNN solution.

### The Reasoning (Flawed)
- **Public Leaderboard Overfitting Concerns**: The competition discussion was filled with warnings about leaderboard overfitting due to the small public test split
- **Last-Minute Decision**: I chose to submit what I thought was a "safer" model

### The Reality
- **Private Leaderboard Results**: The model I submitted performed significantly worse on the private leaderboard
- **Statistical Truth**: In most Kaggle competitions, the discrepancy between public and private leaderboards is typically only 5-10%
- **Lesson Learned**: Strong cross-validation and consistent public performance are usually the best predictors of private performance

### Impact
This single decision transformed what could have been a successful competition result into a disappointing outcome, despite months of dedicated work and model development.

## Umm, Key Learnings and Insights

### Technical Insights
1. **Property-Specific Modeling**: Different polymer properties benefit from different neural network architectures
2. **Data Augmentation Value**: For properties with limited data (like Tc), augmentation techniques are crucial
3. **Ensemble Benefits**: Combining specialized models for different properties improves overall performance

### Competition Strategy Insights
1. **Trust Your CV**: Strong CV performance is usually the best predictor of final performance
2. **Avoid Last-Minute Changes**: Stick to your best-validated approach rather than making strategy changes under pressure
3. **Discussion Forum Caution**: While community discussions provide valuable insights, they can also lead to overthinking and poor decisions
4. **Overfitting Paranoia**: Fear of public leaderboard overfitting can be more harmful than the overfitting itself

## Conclusion

This competition was both a technical challenge and a lesson in decision-making under pressure. 

While my GNN ensemble achieved strong performance (0.065 wMAE), the final submission mistake serves as a reminder that technical excellence must be paired with sound strategic judgment.

The multi-model approach proved effective, with each specialized GNN architecture capturing different aspects of polymer structure-property relationships. The work demonstrates the value of domain-specific model design and the importance of matching model capabilities to problem characteristics.

Despite the disappointing final result, this project significantly advanced my understanding of:
- Graph Neural Networks for molecular property prediction
- Polymer chemistry and structure-property relationships  
- Competition strategy and psychological factors in high-stakes decisions
- The value of systematic approaches over last-minute pivots

## Resources and References

### Competition Links
- [NeurIPS Open Polymer Prediction 2025](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025)
- [Competition Kaggle Notebook](https://www.kaggle.com/code/fridaycode/neurips-gnn-models)

### Referenced Repositories
- [masashitsubaki/molecularGNN_smiles](https://github.com/masashitsubaki/molecularGNN_smiles)
- [hkqiu/DataAugmentation4SmallData](https://github.com/hkqiu/DataAugmentation4SmallData)

### Key Papers
- "Compound-protein Interaction Prediction with End-to-end Learning of Neural Networks for Graphs and Sequences" (Tsubaki et al.)
- "Heat-Resistant Polymer Discovery by Utilizing Interpretable Graph Neural Network with Small Data" (Haoke Qiu, Jingying Wang, ...)
---
